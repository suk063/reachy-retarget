"""Run a JSONL job manifest over the persistent worker pods, ``--slots`` jobs per pod.

Each manifest line: ``{"id", "argv": [...], "publish": "<PVC-relative dir>", "timeout_s"?}``.
``argv`` is run as ``python <argv>`` from the extracted release; ``{out}`` and ``{pvc}``
are substituted on the pod. Jobs are launched detached (``setsid``) so a dropped
kubectl stream never kills them, then polled. A job whose receipt already exists in
``runs/pool/<batch>/`` is skipped, so rerunning the same command resumes a batch.
"""
from __future__ import annotations

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from cluster import k8s, runtime

ROOT = Path(__file__).resolve().parents[1]

PREPARE = """set -eu
mkdir -p /tmp/rr2/runtime /tmp/rr2/release /tmp/rr2/jobs
exec 9>/tmp/rr2/prepare.lock
flock 9
if [ ! -x /tmp/rr2/runtime/{rt}/bin/python ]; then
  tar -C /tmp/rr2/runtime -xzf {pvc}/runtime/{rt}.tar.gz
fi
if [ ! -d /tmp/rr2/release/{rel} ]; then
  mkdir -p /tmp/rr2/release/{rel}.partial
  tar -C /tmp/rr2/release/{rel}.partial -xf {pvc}/releases/{rel}.tar
  mv /tmp/rr2/release/{rel}.partial /tmp/rr2/release/{rel}
fi
"""

LAUNCH = """set -eu
job=/tmp/rr2/jobs/{id}
if [ -e "$job/status.json" ]; then echo exists; exit 0; fi
mkdir -p "$job"
cat > "$job/spec.json"
cd /tmp/rr2/release/{rel}
setsid nohup /tmp/rr2/runtime/{rt}/bin/python -m cluster.job "$job" > "$job/runner.txt" 2>&1 < /dev/null &
echo started
"""

POLL = """for id in {ids}; do
  printf '%s\\t' "$id"; tr -d '\\n' < /tmp/rr2/jobs/$id/status.json 2>/dev/null || printf '{{}}'; echo
done
"""


class Pool:
    def __init__(self, batch, jobs, pods, slots, rt, rel):
        self.batch, self.rt, self.rel = batch, rt, rel
        self.out = ROOT / "runs" / "pool" / batch
        self.out.mkdir(parents=True, exist_ok=True)
        self.todo = [j for j in jobs if not (self.out / f"{j['id']}.json").exists()]
        self.free = [(pod, s) for s in range(slots) for pod in pods]
        random.shuffle(self.free)
        self.running = {}  # id -> (pod, slot, started)
        self.prepared = set()
        self.lock = threading.Lock()
        self.events = (self.out / "events.jsonl").open("a")

    def log(self, **event):
        with self.lock:
            self.events.write(json.dumps({"t": time.time(), **event}) + "\n")
            self.events.flush()

    def launch(self, job, pod, slot):
        try:
            if pod not in self.prepared:
                k8s.run(pod, PREPARE.format(rt=self.rt, rel=self.rel, pvc=k8s.PVC), timeout=900)
                self.prepared.add(pod)
            spec = {**job, "batch": self.batch, "release": self.rel, "slot": slot}
            k8s.run(pod, LAUNCH.format(id=job["id"], rel=self.rel, rt=self.rt),
                    stdin=json.dumps(spec).encode(), timeout=120)
            with self.lock:
                self.running[job["id"]] = (pod, slot, time.time())
            self.log(event="launched", id=job["id"], pod=pod, slot=slot)
        except Exception as error:  # noqa: BLE001 - kubectl failures are retried elsewhere
            self.log(event="launch_error", id=job["id"], pod=pod, error=str(error)[-500:])
            with self.lock:
                self.todo.append(job)
                self.free.append((pod, slot))

    def poll(self, pod, ids):
        try:
            text = k8s.run(pod, POLL.format(ids=" ".join(ids)), timeout=120)
        except Exception as error:  # noqa: BLE001
            self.log(event="poll_error", pod=pod, error=str(error)[-500:])
            return
        for line in text.splitlines():
            job_id, _, payload = line.partition("\t")
            status = json.loads(payload or "{}")
            if status.get("state") in (None, "running"):
                continue
            (self.out / f"{job_id}.json").write_text(json.dumps(status, indent=1, sort_keys=True))
            with self.lock:
                pod_, slot, _ = self.running.pop(job_id)
                self.free.append((pod_, slot))
            self.log(event="finished", id=job_id, pod=pod, state=status["state"])

    def run(self, interval=15.0):
        with ThreadPoolExecutor(32) as executor:
            while self.todo or self.running:
                with self.lock:
                    launches = []
                    while self.todo and self.free:
                        launches.append((self.todo.pop(0), *self.free.pop()))
                for job, pod, slot in launches:
                    executor.submit(self.launch, job, pod, slot)
                time.sleep(interval)
                by_pod = {}
                with self.lock:
                    for job_id, (pod, _, _) in self.running.items():
                        by_pod.setdefault(pod, []).append(job_id)
                list(executor.map(lambda item: self.poll(*item), by_pod.items()))
                done = len(list(self.out.glob("*.json")))
                print(f"{time.strftime('%H:%M:%S')} todo={len(self.todo)} running={len(self.running)} done={done}",
                      flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--batch", help="batch name (default: manifest stem)")
    parser.add_argument("--slots", type=int, default=2, help="concurrent jobs per pod")
    parser.add_argument("--max-pods", type=int)
    parser.add_argument("--exclude-pod", action="append", default=[])
    args = parser.parse_args()
    jobs = [json.loads(line) for line in Path(args.manifest).read_text().splitlines() if line.strip()]
    if len({j["id"] for j in jobs}) != len(jobs):
        raise ValueError("duplicate job ids in manifest")
    pods = [p for p in k8s.ready_pods() if p not in args.exclude_pod][: args.max_pods]
    rt = runtime.build_runtime(pods[0])
    rel = runtime.publish_release(pods[0])
    print(f"batch={args.batch or Path(args.manifest).stem} jobs={len(jobs)} pods={len(pods)} "
          f"slots={args.slots} runtime={rt} release={rel[:12]}", flush=True)
    Pool(args.batch or Path(args.manifest).stem, jobs, pods, args.slots, rt, rel).run()


if __name__ == "__main__":
    main()
