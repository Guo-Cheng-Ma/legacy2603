from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


@dataclass
class KVCacheStats:
    hits: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    misses: int = 0
    inserts: int = 0
    evictions: int = 0
    l1_to_l2: int = 0
    l2_to_l3: int = 0
    l3_drops: int = 0


class KVTier(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


class KVCacheManager:
    """
    Global hash-id KV cache with byte-capacity accounting and per-tier LRU tables.

    C25 keeps APIs compatible with existing single-pool behavior. Runtime transition
    policies are integrated in later commits.
    """

    def __init__(
        self,
        capacity_bytes: int,
        kv_bytes_per_block: int,
        l1_capacity_bytes: Optional[int] = None,
        l2_capacity_bytes: int = 0,
        l3_capacity_bytes: int = 0,
    ):
        if capacity_bytes < 0 and l1_capacity_bytes is None:
            raise ValueError("capacity_bytes must be >= 0 when l1 capacity is not set")
        if kv_bytes_per_block <= 0:
            raise ValueError("kv_bytes_per_block must be > 0")

        l1_capacity = int(capacity_bytes) if l1_capacity_bytes is None else int(l1_capacity_bytes)
        if l1_capacity < 0 or int(l2_capacity_bytes) < 0 or int(l3_capacity_bytes) < 0:
            raise ValueError("tier capacities must be >= 0")

        self.kv_bytes_per_block = int(kv_bytes_per_block)
        self._capacity: Dict[KVTier, int] = {
            KVTier.L1: l1_capacity,
            KVTier.L2: int(l2_capacity_bytes),
            KVTier.L3: int(l3_capacity_bytes),
        }
        self._tiers: Dict[KVTier, OrderedDict] = {
            KVTier.L1: OrderedDict(),
            KVTier.L2: OrderedDict(),
            KVTier.L3: OrderedDict(),
        }
        self._used: Dict[KVTier, int] = {
            KVTier.L1: 0,
            KVTier.L2: 0,
            KVTier.L3: 0,
        }

        self.capacity_bytes = sum(self._capacity.values())
        self.stats = KVCacheStats()

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    @property
    def used_bytes(self) -> int:
        return sum(self._used.values())

    def tier_used_bytes(self, tier: KVTier) -> int:
        return self._used[tier]

    def tier_capacity_bytes(self, tier: KVTier) -> int:
        return self._capacity[tier]

    def locate(self, hash_id: int) -> Optional[KVTier]:
        for tier in [KVTier.L1, KVTier.L2, KVTier.L3]:
            if hash_id in self._tiers[tier]:
                return tier
        return None

    def probe(self, hash_id: int) -> bool:
        tier = self.locate(hash_id)
        if tier is not None:
            self.stats.hits += 1
            if tier == KVTier.L1:
                self.stats.l1_hits += 1
            elif tier == KVTier.L2:
                self.stats.l2_hits += 1
            else:
                self.stats.l3_hits += 1
            self._tiers[tier].move_to_end(hash_id)
            return True
        self.stats.misses += 1
        return False

    def touch(self, hash_id: int) -> bool:
        tier = self.locate(hash_id)
        if tier is None:
            return False
        self._tiers[tier].move_to_end(hash_id)
        return True

    def has(self, hash_id: int) -> bool:
        return self.locate(hash_id) is not None

    def reserve_or_evict(self, required_bytes: int, tier: KVTier = KVTier.L1) -> int:
        """Evict within one tier until required bytes can be allocated. Returns evicted count."""
        if required_bytes <= 0:
            return 0
        if required_bytes > self._capacity[tier]:
            return -1

        evicted = 0
        entries = self._tiers[tier]
        while self._used[tier] + required_bytes > self._capacity[tier] and entries:
            _, _ = entries.popitem(last=False)
            self._used[tier] -= self.kv_bytes_per_block
            self.stats.evictions += 1
            if tier == KVTier.L3:
                self.stats.l3_drops += 1
            evicted += 1
        if self._used[tier] + required_bytes > self._capacity[tier]:
            return -1
        return evicted

    def insert(self, hash_id: int, tier: KVTier = KVTier.L1) -> bool:
        current_tier = self.locate(hash_id)
        if current_tier is not None:
            self._tiers[current_tier].move_to_end(hash_id)
            return True

        result = self.reserve_or_evict(self.kv_bytes_per_block, tier=tier)
        if result < 0:
            return False

        self._tiers[tier][hash_id] = True
        self._used[tier] += self.kv_bytes_per_block
        self.stats.inserts += 1
        return True

    def snapshot(self) -> dict:
        return {
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "num_blocks": sum(len(v) for v in self._tiers.values()),
            "l1_num_blocks": len(self._tiers[KVTier.L1]),
            "l2_num_blocks": len(self._tiers[KVTier.L2]),
            "l3_num_blocks": len(self._tiers[KVTier.L3]),
            "l1_capacity_bytes": self._capacity[KVTier.L1],
            "l2_capacity_bytes": self._capacity[KVTier.L2],
            "l3_capacity_bytes": self._capacity[KVTier.L3],
            "l1_used_bytes": self._used[KVTier.L1],
            "l2_used_bytes": self._used[KVTier.L2],
            "l3_used_bytes": self._used[KVTier.L3],
            "hits": self.stats.hits,
            "l1_hits": self.stats.l1_hits,
            "l2_hits": self.stats.l2_hits,
            "l3_hits": self.stats.l3_hits,
            "misses": self.stats.misses,
            "inserts": self.stats.inserts,
            "evictions": self.stats.evictions,
            "l1_to_l2": self.stats.l1_to_l2,
            "l2_to_l3": self.stats.l2_to_l3,
            "l3_drops": self.stats.l3_drops,
        }
