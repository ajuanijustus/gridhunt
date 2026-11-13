"""Session logs for human-agent teaming studies.

A session writes two files to its directory:

- ``<name>.jsonl``: one JSON record per line, written and flushed as it happens, so a crash
  loses at most the record being written;
- ``<name>_meta.json``: participant, condition, seeds, software versions and environment
  settings, rewritten when the session ends.

Every record has ``type``, ``t`` (seconds since the session started, from a monotonic clock,
for reaction times and for aligning with other recordings) and ``wall_time`` (Unix time).
Record types: ``session_start``, ``episode_start``, ``step``, ``event``, ``questionnaire``,
``episode_end`` and ``session_end``; see the documentation for their fields.
"""

import datetime
import importlib.metadata
import json
import platform
import re
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1


def to_json(value):
    """Convert numpy values, tuples and sets into plain JSON types."""
    if isinstance(value, dict):
        return {str(key): to_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(to_json(item) for item in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _safe(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(text)).strip("-") or "x"


def _versions():
    from .. import __version__

    # The version of the gridhunt code actually imported (an installed copy may differ)
    versions = {"python": platform.python_version(), "platform": platform.platform(), "gridhunt": __version__}
    for distribution in ("numpy", "pettingzoo", "gymnasium", "pygame-ce", "pygame-gui"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


class SessionLog:
    """Writes one participant's session to JSON Lines.

    Args:
        directory: folder for the log files (created if needed).
        participant: participant id, used in the file name.
        condition: experimental condition, stored in every file.
        metadata: any extra information to store in the metadata file.
        log_state: also store ``env.state()`` (the global state vector) on every step.
        on_record: called with every record after it is written, for example to forward
            events to a lab streaming or eye-tracking system.
    """

    def __init__(self, directory, participant="anonymous", condition=None, metadata=None,
                 log_state=True, on_record=None):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.participant = participant
        self.condition = condition
        self.log_state = log_state
        self.on_record = on_record
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{_safe(participant)}_{_safe(condition)}_{stamp}" if condition else f"{_safe(participant)}_{stamp}"
        self.path = self._unique(self.directory / f"{base}.jsonl")
        self.meta_path = self.path.with_name(self.path.stem + "_meta.json")
        self.metadata = {
            "schema_version": SCHEMA_VERSION,
            "participant": participant,
            "condition": condition,
            "log_file": self.path.name,
            "started": datetime.datetime.now().astimezone().isoformat(),
            "versions": _versions(),
            "episodes": [],
            **(metadata or {}),
        }
        self._file = open(self.path, "w", encoding="utf-8")  # noqa: SIM115 (kept open for the session)
        self._t0 = time.perf_counter()
        self._closed = False

    @staticmethod
    def _unique(path):
        candidate, n = path, 1
        while candidate.exists():
            candidate = path.with_name(f"{path.stem}-{n}{path.suffix}")
            n += 1
        return candidate

    def now(self):
        """Seconds since the session started (monotonic clock)."""
        return time.perf_counter() - self._t0

    def write(self, record_type, **fields):
        """Write one record and return it."""
        record = {"type": record_type, "t": round(self.now(), 6), "wall_time": time.time(), **to_json(fields)}
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        if self.on_record is not None:
            self.on_record(record)
        return record

    def update_metadata(self, **fields):
        self.metadata.update(to_json(fields))
        self._write_metadata()

    def _write_metadata(self):
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def close(self):
        if self._closed:
            return
        self.metadata["ended"] = datetime.datetime.now().astimezone().isoformat()
        self._write_metadata()
        self._file.close()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_log(path):
    """Read a session's JSON Lines log into a list of records."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
