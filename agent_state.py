"""Locked, atomic persistence for agent campaigns."""
from contextlib import contextmanager
import fcntl
import json
import os
import re
from pathlib import Path


SCHEMA_VERSION = 1
CAMPAIGN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


class CampaignLockedError(RuntimeError):
    pass


class CampaignStateError(ValueError):
    pass


class CampaignStore:
    def __init__(self, runs_root, campaign_id):
        if not isinstance(campaign_id, str) or not CAMPAIGN_ID_RE.fullmatch(campaign_id):
            raise ValueError("campaign_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}")
        self.runs_root = Path(runs_root)
        self.campaign_id = campaign_id
        self.campaign_dir = self.runs_root / campaign_id
        self.state_path = self.campaign_dir / "state.json"
        self.lock_path = self.runs_root / ".locks" / f"{campaign_id}.lock"
        self._lock_handle = None

    def _reject_symlinks(self):
        for path in (self.runs_root, self.campaign_dir):
            if path.is_symlink():
                raise CampaignStateError(f"Campaign path may not be a symlink: {path}")

    @contextmanager
    def lock(self):
        if self._lock_handle is not None:
            raise CampaignStateError("Campaign lock is already held by this store")
        self._reject_symlinks()
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lock_path.is_symlink():
            raise CampaignStateError(f"Campaign lock may not be a symlink: {self.lock_path}")
        handle = self.lock_path.open("a+")
        try:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise CampaignLockedError(f"Campaign {self.campaign_id!r} is already running") from None
            self._lock_handle = handle
            self._reject_symlinks()
            yield self
        finally:
            self._lock_handle = None
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()

    def _require_lock(self):
        if self._lock_handle is None:
            raise CampaignStateError("Campaign lock must be held for state changes")

    @staticmethod
    def _atomic_write(path, payload):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
        try:
            with temporary.open("wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def exists(self):
        return self.state_path.is_file()

    def initialize(self, state):
        self._require_lock()
        if self.state_path.exists() or self.campaign_dir.exists():
            raise CampaignStateError(
                f"Campaign {self.campaign_id!r} already exists; use agent.py resume")
        value = dict(state)
        value["schema_version"] = SCHEMA_VERSION
        value["campaign_id"] = self.campaign_id
        self.save(value)
        return value

    def load(self):
        try:
            with self.state_path.open(encoding="utf-8") as handle:
                state = json.load(handle)
        except FileNotFoundError:
            raise CampaignStateError(f"Campaign {self.campaign_id!r} does not exist") from None
        except json.JSONDecodeError as exc:
            raise CampaignStateError(f"Campaign state is invalid JSON: {exc}") from None
        if not isinstance(state, dict) or state.get("schema_version") != SCHEMA_VERSION:
            raise CampaignStateError("Unsupported or missing campaign state schema version")
        if state.get("campaign_id") != self.campaign_id:
            raise CampaignStateError("Campaign state ID does not match its directory")
        return state

    def save(self, state):
        self._require_lock()
        if state.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
            raise CampaignStateError("Unsupported campaign state schema version")
        if state.get("campaign_id", self.campaign_id) != self.campaign_id:
            raise CampaignStateError("Campaign state ID cannot change")
        value = dict(state)
        value["schema_version"] = SCHEMA_VERSION
        value["campaign_id"] = self.campaign_id
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        self._atomic_write(self.state_path, payload)

    def iteration_dir(self, iteration):
        if not isinstance(iteration, int) or iteration <= 0:
            raise ValueError("iteration must be a positive integer")
        return self.campaign_dir / "iterations" / f"{iteration:04d}"

    def write_json(self, path, value):
        self._require_lock()
        payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
        self._atomic_write(path, payload)

    def write_bytes(self, path, value):
        self._require_lock()
        if not isinstance(value, bytes):
            raise TypeError("value must be bytes")
        self._atomic_write(path, value)

    def write_text(self, path, value):
        self.write_bytes(path, value.encode("utf-8"))
