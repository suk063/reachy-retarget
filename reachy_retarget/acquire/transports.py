"""Byte transports shared by every acquisition kind (no network on import).

An *opener* is any callable ``opener(urllib.request.Request) -> response`` with the
``urllib.request.urlopen`` interface (``read``, ``status``, ``headers``, context
manager). Tests pass in-process fakes; :func:`default_opener` is ``urlopen`` and
:class:`KeepAliveOpener` reuses HTTPS connections and caches redirect targets for
many small range reads of the same file.

* :class:`ResumingStream` - sequential, hashing (SHA-1 + SHA-256) reader of a whole URL
  that resumes a dropped connection with HTTP ``Range``.
* :func:`read_range` / :func:`range_chunks` - one byte range, refusing a server that ignores
  the range (a 200 response would transfer a whole archive).
* :class:`RangeFile` - seekable read-only file over HTTP ``Range`` (zip central directories).
* :func:`zstd_stream` - a readable stream of zstd-decompressed chunks (``zstandard`` or the
  ``zstd`` CLI).
* :func:`parse_tar_header` / :func:`walk_tar` - tar member headers read by small ranges.
* :func:`is_image_name` - the image/video guard applied to every file name written.
"""
from __future__ import annotations

import hashlib
import http.client
import io
import shutil
import subprocess
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

CHUNK = 1 << 20
USER_AGENT = "reachy-retarget-fetch/3"
IMAGE_EXT = (".mp4", ".avi", ".mkv", ".webm", ".mov", ".m4v", ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif",
             ".tiff", ".exr", ".webp", ".heic")
IMAGE_DIRS = ("videos", "images")


def is_image_name(name: str) -> bool:
    """True for a path under ``videos/``/``images/`` or with an image/video extension."""
    parts = name.replace("\\", "/").split("/")
    return any(p in IMAGE_DIRS for p in parts[:-1]) or name.lower().endswith(IMAGE_EXT)


def default_opener(req):
    return urllib.request.urlopen(req, timeout=120)


def make_request(url: str, start: int | None = None, end: int | None = None) -> urllib.request.Request:
    """GET ``url``; ``start``/``end`` (inclusive) add a ``Range`` header."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    if start is not None:
        req.add_header("Range", f"bytes={start}-" + ("" if end is None else str(end)))
    return req


class RangeIgnored(RuntimeError):
    pass


def _status(resp) -> int:
    return getattr(resp, "status", None) or 200


def read_range(url: str, offset: int, size: int, opener, retries: int = 4) -> bytes:
    """Bytes ``[offset, offset + size)`` of ``url``; refuses a server that ignores ``Range``."""
    for attempt in range(retries + 1):
        try:
            with opener(make_request(url, offset, offset + size - 1)) as resp:
                if _status(resp) != 206:
                    raise RangeIgnored(f"{url}: server ignored the byte range (status {_status(resp)}); "
                                       "refusing to read a whole archive")
                data = resp.read()
            break
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            if attempt == retries or (isinstance(exc, urllib.error.HTTPError) and exc.code in (404, 416)):
                raise
            time.sleep(min(30, 2 ** attempt))
    if len(data) != size:
        raise IOError(f"{url}: got {len(data)} bytes for range [{offset}, {offset + size}), expected {size}")
    return data


def range_chunks(url: str, offset: int, size: int, opener, digest=None, counter=None):
    """Yield the bytes ``[offset, offset + size)`` in chunks, hashing them into ``digest``."""
    with opener(make_request(url, offset, offset + size - 1)) as resp:
        if _status(resp) != 206:
            raise RangeIgnored(f"{url}: server ignored the byte range (status {_status(resp)}); "
                               "refusing to stream a whole archive")
        for block in iter(lambda: resp.read(CHUNK), b""):
            if digest is not None:
                digest.update(block)
            if counter is not None:
                counter[0] += len(block)
            yield block


class ResumingStream(io.RawIOBase):
    """Sequential reader of ``url`` that hashes every byte and resumes with ``Range``.

    ``start`` > 0 asks for the bytes from ``start`` on; if the server answers the first
    request with the whole file instead, the stream restarts at 0 and ``restarted`` is set.
    """

    def __init__(self, url, opener, retries=8, start=0):
        self.url, self.opener, self.retries = url, opener, retries
        self.pos, self.total, self.restarted = start, None, False
        self.sha1, self.sha256 = hashlib.sha1(), hashlib.sha256()
        self.resp = None
        self._open(first=True)

    def _open(self, first=False):
        self.resp = self.opener(make_request(self.url, self.pos or None))
        status = _status(self.resp)
        if self.pos and status != 206:
            if not first:
                raise RangeIgnored(f"{self.url}: server ignored Range on resume (status {status})")
            self.pos, self.restarted = 0, True
        headers = getattr(self.resp, "headers", None) or {}
        rng, length = headers.get("Content-Range"), headers.get("Content-Length")
        if rng:
            self.total = int(rng.rsplit("/", 1)[1])
        elif length and not self.pos:
            self.total = int(length)

    def readable(self):
        return True

    def readinto(self, b):
        for attempt in range(self.retries + 1):
            try:
                n = self.resp.readinto(b) if hasattr(self.resp, "readinto") else None
                if n is None:
                    data = self.resp.read(len(b))
                    n = len(data)
                    b[:n] = data
                break
            except (OSError, urllib.error.URLError, http.client.HTTPException):
                if attempt == self.retries:
                    raise
                try:
                    self.resp.close()
                finally:
                    time.sleep(min(30, 2 ** attempt))
                    self._open()
        view = memoryview(b)[:n]
        self.sha1.update(view)
        self.sha256.update(view)
        self.pos += n
        return n

    def drain(self):
        buf = bytearray(CHUNK)
        while self.readinto(buf):
            pass

    def close(self):
        if self.resp is not None:
            self.resp.close()
        super().close()


class RangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file through HTTP ``Range`` requests."""

    def __init__(self, url, opener, size: int | None = None):
        self.url, self.opener, self.pos = url, opener, 0
        if size is None:
            with opener(make_request(url, 0, 0)) as r:
                self.url = r.geturl() if hasattr(r, "geturl") else url
                size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
        self.size = size

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = read_range(self.url, self.pos, n, self.opener)
        b[:n] = data
        self.pos += n
        return n


# ---------------------------------------------------------------- keep-alive opener

class _KAResponse(io.RawIOBase):
    def __init__(self, resp, url, release):
        self._resp, self._url, self._release = resp, url, release
        self.status, self.headers = resp.status, resp.headers

    def readable(self):
        return True

    def readinto(self, b):
        return self._resp.readinto(b)

    def read(self, n=-1):
        return self._resp.read() if n is None or n < 0 else self._resp.read(n)

    def geturl(self):
        return self._url

    def close(self):
        if not self.closed:
            self._release(self._resp)
        super().close()


class KeepAliveOpener:
    """``urlopen``-compatible opener with per-thread persistent HTTPS connections.

    Redirect targets of a URL (e.g. signed CDN URLs) are cached and reused until the
    target answers 401/403/404/410, then resolved again. 429/503 answers are retried
    after ``Retry-After`` (or an exponential back-off). Used for many small ranges of
    one archive; behaves like ``urlopen`` for the request types used here (GET, Range).
    """

    def __init__(self, timeout=120, max_redirects=5, max_throttle_retries=8):
        self.timeout, self.max_redirects = timeout, max_redirects
        self.max_throttle_retries = max_throttle_retries
        self._local = threading.local()
        self._redirects: dict[str, str] = {}
        self._lock = threading.Lock()

    def _conn(self, scheme, netloc):
        pool = getattr(self._local, "pool", None)
        if pool is None:
            pool = self._local.pool = {}
        key = (scheme, netloc)
        if key not in pool:
            cls = http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection
            pool[key] = cls(netloc, timeout=self.timeout)
        return key, pool[key]

    def _drop(self, key):
        conn = self._local.pool.pop(key, None)
        if conn is not None:
            conn.close()

    def _once(self, url, headers):
        u = urllib.parse.urlsplit(url)
        path = u.path + (f"?{u.query}" if u.query else "")
        for attempt in range(2):  # a stale keep-alive connection is retried once
            key, conn = self._conn(u.scheme, u.netloc)
            try:
                conn.request("GET", path, headers=headers)
                return key, conn.getresponse()
            except (http.client.HTTPException, OSError):
                self._drop(key)
                if attempt:
                    raise

    def __call__(self, req):
        orig = req.full_url
        headers = {k: v for k, v in req.header_items()}
        with self._lock:
            url = self._redirects.get(orig, orig)
        throttled = 0
        for _ in range(self.max_redirects + 1 + self.max_throttle_retries):
            key, resp = self._once(url, headers)
            if resp.status in (429, 503) and throttled < self.max_throttle_retries:
                # Rate limited: honour Retry-After (seconds), else back off exponentially.
                resp.read()
                retry_after = resp.getheader("Retry-After") or ""
                time.sleep(min(300.0, float(retry_after) if retry_after.isdigit() else 5.0 * 2 ** throttled))
                throttled += 1
                continue
            if resp.status in (301, 302, 303, 307, 308):
                loc = urllib.parse.urljoin(url, resp.getheader("Location"))
                resp.read()
                url = loc
                continue
            if resp.status in (401, 403, 404, 410) and url != orig:
                resp.read()
                with self._lock:
                    self._redirects.pop(orig, None)
                url = orig
                continue
            if resp.status >= 400:
                body = resp.read()
                raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, io.BytesIO(body))
            if url != orig:
                with self._lock:
                    self._redirects[orig] = url

            def release(r, key=key):
                try:
                    r.read()  # leave the connection reusable
                except (http.client.HTTPException, OSError):
                    self._drop(key)
            return _KAResponse(resp, url, release)
        raise urllib.error.URLError(f"{orig}: too many redirects")


# ---------------------------------------------------------------- zstd

class _ChunkReader(io.RawIOBase):
    def __init__(self, it):
        self.it, self.buf = it, b""

    def readable(self):
        return True

    def readinto(self, b):
        while not self.buf:
            try:
                self.buf = next(self.it)
            except StopIteration:
                return 0
        n = min(len(b), len(self.buf))
        b[:n], self.buf = self.buf[:n], self.buf[n:]
        return n


def zstd_stream(chunks):
    """``(stream, finish)``: a readable stream of the zstd-decompressed ``chunks``.

    ``finish()`` waits for the decompressor and raises its error; ``finish(abort=True)``
    stops it and returns the producer's error (e.g. a refused range), if any.
    """
    try:
        import zstandard  # optional dependency (pyproject extra "archives")
    except ImportError:
        zstandard = None
    if zstandard is not None:
        reader = zstandard.ZstdDecompressor().stream_reader(io.BufferedReader(_ChunkReader(iter(chunks)), CHUNK))
        return reader, lambda abort=False: None
    exe = shutil.which("zstd")
    if exe is None:
        raise RuntimeError("zstd-compressed source: install the 'zstandard' package or the zstd CLI")
    proc = subprocess.Popen([exe, "-dc"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    err = []

    def feed():
        try:
            for c in chunks:
                proc.stdin.write(c)
        except BaseException as exc:  # surfaced by finish()
            err.append(exc)
        finally:
            try:
                proc.stdin.close()
            except OSError:
                pass

    th = threading.Thread(target=feed, daemon=True)
    th.start()

    def finish(abort=False):
        if abort:
            proc.kill()
            proc.stdout.close()
        th.join()
        rc = proc.wait()
        if abort:
            return err[0] if err else None
        if err:
            raise err[0]
        if rc != 0:
            raise RuntimeError(f"zstd failed: {proc.stderr.read().decode(errors='replace')}")
    return proc.stdout, finish


# ---------------------------------------------------------------- tar headers

def parse_tar_header(block: bytes) -> tarfile.TarInfo | None:
    """A valid ustar/GNU header block, or ``None`` (wrong checksum, zero block, garbage)."""
    if len(block) != 512 or block == b"\0" * 512:
        return None
    try:
        return tarfile.TarInfo.frombuf(block, "utf-8", "surrogateescape")
    except tarfile.HeaderError:
        return None


def check_tar_header(block: bytes, name: str, size: int) -> None:
    """``block`` is the header of a regular member of ``size`` bytes whose name ends ``name``'s
    stored prefix (ustar names longer than 100 bytes are stored in a preceding record)."""
    ti = parse_tar_header(block)
    if ti is None or not (ti.isfile() or ti.type == tarfile.AREGTYPE):
        raise IOError(f"{name}: no tar file header before the member data")
    if ti.size != size:
        raise IOError(f"{name}: tar header size {ti.size} != catalogued {size}")
    stored = ti.name  # long names are truncated in the header that precedes the data
    if not stored or not (name.startswith(stored[:50]) or name.endswith(stored[-50:])):
        raise IOError(f"{name}: tar header names {stored!r}")


def walk_tar(read, start: int = 0, stop: int | None = None, lookahead: int = 0):
    """Yield ``(data_offset, size, name, type)`` of tar members by reading headers only.

    ``read(offset, n)`` returns archive bytes (shorter at the end). ``stop`` ends the walk
    at the first header at or past that offset. ``lookahead`` bytes are read together with
    each header so that small members and the following header need no extra request.
    GNU long names and pax ``path``/``size`` records are honoured.
    """
    off, buf_off, buf = start, 0, b""

    def get(o, n):
        nonlocal buf_off, buf
        if not (buf_off <= o and o + n <= buf_off + len(buf)):
            buf_off, buf = o, read(o, n + lookahead)
        return buf[o - buf_off:o - buf_off + n]

    while stop is None or off < stop:
        h = get(off, 512)
        if len(h) < 512 or h == b"\0" * 512:
            return
        ti = parse_tar_header(h)
        if ti is None:
            raise IOError(f"invalid tar header at offset {off}")
        hdr, name, size, typ = 512, ti.name, ti.size, ti.type
        if typ in (tarfile.GNUTYPE_LONGNAME, tarfile.XHDTYPE, tarfile.XGLTYPE):
            nblk = (ti.size + 511) // 512
            ext = get(off + 512, nblk * 512 + 512)
            real = parse_tar_header(ext[nblk * 512:])
            if real is None:
                raise IOError(f"invalid tar header after extended record at offset {off}")
            name, size, typ = real.name, real.size, real.type
            if ti.type == tarfile.GNUTYPE_LONGNAME:
                name = ext[:ti.size].rstrip(b"\0").decode("utf-8", "surrogateescape")
            elif ti.type == tarfile.XHDTYPE:
                for rec in _pax_records(ext[:ti.size]):
                    if rec[0] == "path":
                        name = rec[1]
                    elif rec[0] == "size":
                        size = int(rec[1])
            hdr = 512 + nblk * 512 + 512
        yield off + hdr, size, name, typ
        off += hdr + ((size + 511) // 512) * 512


def _pax_records(raw: bytes):
    pos = 0
    while pos < len(raw):
        sp = raw.index(b" ", pos)
        length = int(raw[pos:sp])
        key, _, value = raw[sp + 1:pos + length - 1].decode("utf-8", "surrogateescape").partition("=")
        yield key, value
        pos += length


def find_tar_header(data: bytes, base: int) -> int | None:
    """Offset (absolute, ``base`` = offset of ``data``) of the first 512-aligned block in
    ``data`` that parses as a tar header with a valid checksum (archive-relative alignment)."""
    first = (-base) % 512
    for i in range(first, len(data) - 511, 512):
        block = data[i:i + 512]
        if block[257:262] == b"ustar" and parse_tar_header(block) is not None:
            return base + i
    return None


__all__ = ["CHUNK", "USER_AGENT", "IMAGE_EXT", "KeepAliveOpener", "RangeFile", "RangeIgnored", "ResumingStream",
           "check_tar_header", "default_opener", "find_tar_header", "is_image_name", "make_request",
           "parse_tar_header", "range_chunks", "read_range", "walk_tar", "zstd_stream"]
