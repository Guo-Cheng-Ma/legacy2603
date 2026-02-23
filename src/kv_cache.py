from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class KVCacheStats:
    hits: int = 0
    misses: int = 0
    inserts: int = 0
    evictions: int = 0


class KVCacheManager:
    """Global hash-id KV cache with byte-capacity accounting and LRU eviction."""

    def __init__(self, capacity_bytes: int, kv_bytes_per_block: int):
        if capacity_bytes < 0:
            raise ValueError("capacity_bytes must be >= 0")
        if kv_bytes_per_block <= 0:
            raise ValueError("kv_bytes_per_block must be > 0")

        self.capacity_bytes = int(capacity_bytes)
        self.kv_bytes_per_block = int(kv_bytes_per_block)

        self._entries = OrderedDict()
        self.used_bytes = 0
        self.stats = KVCacheStats()

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def probe(self, hash_id: int) -> bool:
        if hash_id in self._entries:
            self.stats.hits += 1
            self._entries.move_to_end(hash_id)
            return True
        self.stats.misses += 1
        return False

    def touch(self, hash_id: int) -> bool:
        if hash_id not in self._entries:
            return False
        self._entries.move_to_end(hash_id)
        return True

    def has(self, hash_id: int) -> bool:
        return hash_id in self._entries

    def reserve_or_evict(self, required_bytes: int) -> int:
        """Evict until required bytes can be allocated. Returns evicted count."""
        if required_bytes <= 0:
            return 0
        if required_bytes > self.capacity_bytes:
            return -1

        evicted = 0
        while self.used_bytes + required_bytes > self.capacity_bytes and self._entries:
            _, _ = self._entries.popitem(last=False)
            self.used_bytes -= self.kv_bytes_per_block
            self.stats.evictions += 1
            evicted += 1
        if self.used_bytes + required_bytes > self.capacity_bytes:
            return -1
        return evicted

    def insert(self, hash_id: int) -> bool:
        if hash_id in self._entries:
            self._entries.move_to_end(hash_id)
            return True

        result = self.reserve_or_evict(self.kv_bytes_per_block)
        if result < 0:
            return False

        self._entries[hash_id] = True
        self.used_bytes += self.kv_bytes_per_block
        self.stats.inserts += 1
        return True

    def snapshot(self) -> dict:
        return {
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "num_blocks": len(self._entries),
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "inserts": self.stats.inserts,
            "evictions": self.stats.evictions,
        }
