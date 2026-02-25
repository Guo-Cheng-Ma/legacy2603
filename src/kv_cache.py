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


@dataclass
class KVAccessResult:
    is_hit: bool = False
    hit_tier: Optional[KVTier] = None
    inserted: bool = False
    promoted_from_l2: int = 0
    promoted_from_l3: int = 0
    l1_to_l2: int = 0
    l2_to_l3: int = 0
    l3_drops: int = 0

    @property
    def l1_hit(self) -> int:
        return 1 if self.hit_tier == KVTier.L1 else 0

    @property
    def l2_hit(self) -> int:
        return 1 if self.hit_tier == KVTier.L2 else 0

    @property
    def l3_hit(self) -> int:
        return 1 if self.hit_tier == KVTier.L3 else 0

    @property
    def dma_blocks(self) -> int:
        # L1<->L2 movement.
        return self.l1_to_l2 + self.promoted_from_l2

    @property
    def pcie_blocks(self) -> int:
        # L2<->L3 and L3->L1 movement.
        return self.l2_to_l3 + self.promoted_from_l3

    @property
    def migration_blocks(self) -> int:
        return self.dma_blocks + self.pcie_blocks


class KVCacheManager:
    """
    Global hash-id KV cache with 3-tier LRU state.

    `access(hash_id)` applies tier transitions and always tries to leave the block in L1.
    """

    def __init__(
        self,
        capacity_bytes: int,
        kv_bytes_per_block: int,
        l1_capacity_bytes: Optional[int] = None,
        l2_capacity_bytes: int = 0,
        l3_capacity_bytes: int = 0,
        spare_ratio: float = 0.01,
    ):
        if capacity_bytes < 0 and l1_capacity_bytes is None:
            raise ValueError("capacity_bytes must be >= 0 when l1 capacity is not set")
        if kv_bytes_per_block <= 0:
            raise ValueError("kv_bytes_per_block must be > 0")

        l1_capacity = int(capacity_bytes) if l1_capacity_bytes is None else int(l1_capacity_bytes)
        if l1_capacity < 0 or int(l2_capacity_bytes) < 0 or int(l3_capacity_bytes) < 0:
            raise ValueError("tier capacities must be >= 0")
        if float(spare_ratio) < 0.0 or float(spare_ratio) >= 1.0:
            raise ValueError("spare_ratio must be in [0.0, 1.0)")

        self.kv_bytes_per_block = int(kv_bytes_per_block)
        self.spare_ratio = float(spare_ratio)
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

    def _effective_capacity(self, tier: KVTier) -> int:
        cap = self._capacity[tier]
        reserve = int(cap * self.spare_ratio)
        threshold = cap - reserve
        if cap >= self.kv_bytes_per_block:
            threshold = max(threshold, self.kv_bytes_per_block)
        return max(threshold, 0)

    def _pop_lru(self, tier: KVTier) -> Optional[int]:
        entries = self._tiers[tier]
        if not entries:
            return None
        hash_id, _ = entries.popitem(last=False)
        self._used[tier] -= self.kv_bytes_per_block
        return hash_id

    def _remove_if_exists(self, hash_id: int, tier: KVTier) -> bool:
        entries = self._tiers[tier]
        if hash_id not in entries:
            return False
        del entries[hash_id]
        self._used[tier] -= self.kv_bytes_per_block
        return True

    def locate(self, hash_id: int) -> Optional[KVTier]:
        for tier in [KVTier.L1, KVTier.L2, KVTier.L3]:
            if hash_id in self._tiers[tier]:
                return tier
        return None

    def _place_in_l3(self, hash_id: int, result: Optional[KVAccessResult] = None) -> bool:
        cap = self._effective_capacity(KVTier.L3)
        if cap < self.kv_bytes_per_block:
            self.stats.evictions += 1
            self.stats.l3_drops += 1
            if result is not None:
                result.l3_drops += 1
            return False

        while self._used[KVTier.L3] + self.kv_bytes_per_block > cap:
            victim = self._pop_lru(KVTier.L3)
            if victim is None:
                break
            self.stats.evictions += 1
            self.stats.l3_drops += 1
            if result is not None:
                result.l3_drops += 1

        if self._used[KVTier.L3] + self.kv_bytes_per_block > cap:
            return False

        self._tiers[KVTier.L3][hash_id] = True
        self._used[KVTier.L3] += self.kv_bytes_per_block
        return True

    def _place_in_l2(self, hash_id: int, result: Optional[KVAccessResult] = None) -> bool:
        cap = self._effective_capacity(KVTier.L2)
        if cap < self.kv_bytes_per_block:
            return self._place_in_l3(hash_id, result=result)

        while self._used[KVTier.L2] + self.kv_bytes_per_block > cap:
            victim = self._pop_lru(KVTier.L2)
            if victim is None:
                break
            self.stats.evictions += 1
            self.stats.l2_to_l3 += 1
            if result is not None:
                result.l2_to_l3 += 1
            self._place_in_l3(victim, result=result)

        if self._used[KVTier.L2] + self.kv_bytes_per_block > cap:
            return self._place_in_l3(hash_id, result=result)

        self._tiers[KVTier.L2][hash_id] = True
        self._used[KVTier.L2] += self.kv_bytes_per_block
        return True

    def _place_in_l1(self, hash_id: int, result: Optional[KVAccessResult] = None) -> bool:
        cap = self._effective_capacity(KVTier.L1)
        if cap < self.kv_bytes_per_block:
            return False

        while self._used[KVTier.L1] + self.kv_bytes_per_block > cap:
            victim = self._pop_lru(KVTier.L1)
            if victim is None:
                break
            self.stats.evictions += 1
            self.stats.l1_to_l2 += 1
            if result is not None:
                result.l1_to_l2 += 1
            self._place_in_l2(victim, result=result)

        if self._used[KVTier.L1] + self.kv_bytes_per_block > cap:
            return False

        self._tiers[KVTier.L1][hash_id] = True
        self._used[KVTier.L1] += self.kv_bytes_per_block
        return True

    def probe(self, hash_id: int) -> bool:
        """Compatibility API: hit test without tier promotions."""
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

    def access(self, hash_id: int) -> KVAccessResult:
        """
        Main API for continuous scheduler.
        - hit in L1: reuse directly.
        - hit in L2/L3: promote to L1 and apply cascading evictions.
        - miss: insert as newly computed block into L1.
        """
        result = KVAccessResult()
        tier = self.locate(hash_id)
        if tier == KVTier.L1:
            self.stats.hits += 1
            self.stats.l1_hits += 1
            result.is_hit = True
            result.hit_tier = KVTier.L1
            self._tiers[KVTier.L1].move_to_end(hash_id)
            return result

        if tier == KVTier.L2:
            self.stats.hits += 1
            self.stats.l2_hits += 1
            result.is_hit = True
            result.hit_tier = KVTier.L2
            self._remove_if_exists(hash_id, KVTier.L2)
            if self._place_in_l1(hash_id, result=result):
                result.promoted_from_l2 = 1
            else:
                self._place_in_l2(hash_id, result=result)
            return result

        if tier == KVTier.L3:
            self.stats.hits += 1
            self.stats.l3_hits += 1
            result.is_hit = True
            result.hit_tier = KVTier.L3
            self._remove_if_exists(hash_id, KVTier.L3)
            if self._place_in_l1(hash_id, result=result):
                result.promoted_from_l3 = 1
            else:
                self._place_in_l3(hash_id, result=result)
            return result

        self.stats.misses += 1
        if self._place_in_l1(hash_id, result=result):
            self.stats.inserts += 1
            result.inserted = True
        return result

    def touch(self, hash_id: int) -> bool:
        tier = self.locate(hash_id)
        if tier is None:
            return False
        self._tiers[tier].move_to_end(hash_id)
        return True

    def has(self, hash_id: int) -> bool:
        return self.locate(hash_id) is not None

    def reserve_or_evict(self, required_bytes: int, tier: KVTier = KVTier.L1) -> int:
        """Compatibility API used by legacy code paths."""
        if required_bytes <= 0:
            return 0
        cap = self._effective_capacity(tier)
        if required_bytes > cap:
            return -1

        evicted = 0
        while self._used[tier] + required_bytes > cap:
            victim = self._pop_lru(tier)
            if victim is None:
                break
            self.stats.evictions += 1
            if tier == KVTier.L3:
                self.stats.l3_drops += 1
            evicted += 1

        if self._used[tier] + required_bytes > cap:
            return -1
        return evicted

    def insert(self, hash_id: int, tier: KVTier = KVTier.L1) -> bool:
        """Compatibility API: explicit placement by tier."""
        current_tier = self.locate(hash_id)
        if current_tier is not None:
            self._tiers[current_tier].move_to_end(hash_id)
            return True

        result = KVAccessResult()
        ok = False
        if tier == KVTier.L1:
            ok = self._place_in_l1(hash_id, result=result)
        elif tier == KVTier.L2:
            ok = self._place_in_l2(hash_id, result=result)
        else:
            ok = self._place_in_l3(hash_id, result=result)

        if ok:
            self.stats.inserts += 1
        return ok

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
            "l1_effective_capacity_bytes": self._effective_capacity(KVTier.L1),
            "l2_effective_capacity_bytes": self._effective_capacity(KVTier.L2),
            "l3_effective_capacity_bytes": self._effective_capacity(KVTier.L3),
            "spare_ratio": self.spare_ratio,
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
