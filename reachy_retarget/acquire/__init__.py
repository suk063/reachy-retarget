"""Pinned, explicit acquisition of source files: ``catalog/<family>.yaml`` + :func:`fetch`.

One interface for every acquisition kind (``file``, ``file_images_embedded``,
``tar_stream``, ``range_member``, ``range_package``; see :mod:`.catalog`), one per-file
ledger (:mod:`.ledger`), shared transports (:mod:`.transports`). Catalog generators that
read publisher metadata live in :mod:`.generators` and use the network only when invoked.
"""
from .catalog import KINDS, CatalogEntry, load_catalog, select
from .fetch import (PACKAGE_MANIFEST, RESERVE_BYTES, TAR_MANIFEST, ChecksumMismatch, ImageRefused,
                    InsufficientDisk, fetch, fetch_one, find_entry, identify, locate_entry, sha256_file)
from .ledger import (ledger_record_path, migrate_ledger, read_ledger, read_ledger_record, record_digest,
                     same_record)
from .strip import STRIP_ATTR, read_strip_record, strip_images, stripped_path
from .transports import is_image_name

__all__ = ["KINDS", "PACKAGE_MANIFEST", "RESERVE_BYTES", "STRIP_ATTR", "TAR_MANIFEST", "CatalogEntry",
           "ChecksumMismatch", "ImageRefused", "InsufficientDisk", "fetch", "fetch_one", "find_entry", "identify",
           "is_image_name", "ledger_record_path", "load_catalog", "locate_entry", "migrate_ledger", "read_ledger",
           "read_ledger_record", "read_strip_record", "record_digest", "same_record", "select", "sha256_file",
           "strip_images", "stripped_path"]
