"""Append-only, file-locked JSONL cache for annotation votes.

Ported from `inverse_cai.algorithm.cache` (inverse-cai v0.4.8). Allows interrupted
annotation runs to be resumed when re-run with the same output directory.
"""

import hashlib
import json
import time
from pathlib import Path

from filelock import FileLock
from loguru import logger


class VoteCache:
    def __init__(
        self,
        cache_path: Path,
        lock_timeout: int = 10,
        max_entries_per_file: int = 10000,
    ):
        """Initialize vote cache.

        Args:
            cache_path: Base path of the cache files (".jsonl" suffix is added,
                together with a file index for rotation)
            lock_timeout: Maximum time to wait for file lock in seconds
            max_entries_per_file: Maximum entries per cache file before rotation
        """
        self.cache_path = Path(cache_path)
        self.cache_dir = self.cache_path.parent
        self.cache_base = self.cache_path.name
        self.cache_ext = ".jsonl"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(
            self.cache_dir / f"{self.cache_base}.lock", timeout=lock_timeout
        )
        self.max_entries_per_file = max_entries_per_file

        self._processed_hashes = None

        # Find the active (highest numbered) cache file and count its entries
        file_index = 0
        while self._get_cache_file_path(file_index).exists():
            file_index += 1
        self._current_file_index = max(0, file_index - 1)
        current_cache_path = self._get_cache_file_path(self._current_file_index)
        if current_cache_path.exists():
            with open(current_cache_path, "r", encoding="utf-8") as f:
                self._current_file_entries = sum(1 for line in f if line.strip())
        else:
            self._current_file_entries = 0
            current_cache_path.touch()

    def _get_cache_file_path(self, file_index: int) -> Path:
        return self.cache_dir / f"{self.cache_base}_{file_index}{self.cache_ext}"

    def _iter_cache_data(self):
        file_index = 0
        while True:
            cache_file = self._get_cache_file_path(file_index)
            if not cache_file.exists():
                break
            with open(cache_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        try:
                            yield json.loads(line.strip())
                        except json.JSONDecodeError as e:
                            logger.warning(
                                f"Skipping malformed line in cache file {cache_file}: {line.strip()}, error: {e}"
                            )
            file_index += 1

    def get_cached_votes(self) -> dict:
        """Get all cached votes as dictionary of hash -> vote."""
        with self.lock:
            votes_dict = {}
            for data in self._iter_cache_data():
                votes_dict.update(data)
            self._processed_hashes = set(votes_dict.keys())
            return votes_dict

    def update_cache(self, hash_str: str, vote) -> None:
        """Append a new vote to the cache (no-op if hash already cached)."""
        if self._processed_hashes is None:
            self.get_cached_votes()

        max_retries = 10
        for attempt in range(max_retries):
            try:
                with self.lock:
                    if hash_str in self._processed_hashes:
                        return
                    if self._current_file_entries >= self.max_entries_per_file:
                        self._current_file_index += 1
                        self._current_file_entries = 0
                    with open(
                        self._get_cache_file_path(self._current_file_index),
                        "a",
                        encoding="utf-8",
                    ) as f:
                        json.dump({hash_str: vote}, f)
                        f.write("\n")
                    self._processed_hashes.add(hash_str)
                    self._current_file_entries += 1
                return
            except TimeoutError as exc:
                if attempt < max_retries - 1:
                    logger.warning("Cache lock acquisition failed, retrying in 1s...")
                    time.sleep(1)
                else:
                    raise TimeoutError(
                        f"Failed to acquire cache lock after {max_retries} retries"
                    ) from exc


def _get_hash(dict_to_hash: dict) -> str:
    string = json.dumps(dict_to_hash, sort_keys=True)
    return hashlib.md5(string.encode("utf-8")).hexdigest()[:8]


def get_vote_hash(
    preferred: str,
    rejected: str,
    principle: str,
    model_name: str,
    shown_first: str | None = None,
) -> str:
    """Hash identifying a single vote.

    Args:
        shown_first: For position-debiased annotation, which response was shown
            as sample A ("preferred" or "rejected"). None for standard annotation
            (in which case the hash is identical to ICAI's vote hash).
    """
    to_hash = dict(
        preferred=preferred,
        rejected=rejected,
        principle=principle,
        model_name=model_name,
    )
    if shown_first is not None:
        to_hash["shown_first"] = shown_first
    return _get_hash(to_hash)
