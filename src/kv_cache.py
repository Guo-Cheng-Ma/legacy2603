from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional, Tuple


class KVTier(str, Enum):
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"


@dataclass(frozen=True)
class KVLocation:
    tier: KVTier
    card: Optional[int] = None
    die: Optional[int] = None
    bank: Optional[int] = None


@dataclass
class KVTransferDomainDemands:
    dma: Dict[Tuple[int, int], float] = field(default_factory=dict)
    hbm_read: Dict[Tuple[int, int], float] = field(default_factory=dict)
    hbm_write: Dict[Tuple[int, int], float] = field(default_factory=dict)
    nvlink: Dict[Tuple[int, int], float] = field(default_factory=dict)
    pcie: Dict[int, float] = field(default_factory=dict)

    @staticmethod
    def _merge_time(mapping: Dict, key, time_s: float) -> None:
        time_s = float(time_s)
        if time_s <= 0.0 or key is None:
            return
        mapping[key] = mapping.get(key, 0.0) + time_s

    def add_dma(self, card: Optional[int], die: Optional[int], time_s: float) -> None:
        if card is None or die is None:
            return
        self._merge_time(self.dma, (int(card), int(die)), time_s)

    def add_hbm_read(self, card: Optional[int], die: Optional[int], time_s: float) -> None:
        if card is None or die is None:
            return
        self._merge_time(self.hbm_read, (int(card), int(die)), time_s)

    def add_hbm_write(self, card: Optional[int], die: Optional[int], time_s: float) -> None:
        if card is None or die is None:
            return
        self._merge_time(self.hbm_write, (int(card), int(die)), time_s)

    def add_nvlink(self, src_card: Optional[int], dst_card: Optional[int], time_s: float) -> None:
        if src_card is None or dst_card is None:
            return
        self._merge_time(self.nvlink, (int(src_card), int(dst_card)), time_s)

    def add_pcie(self, card: Optional[int], time_s: float) -> None:
        if card is None:
            return
        self._merge_time(self.pcie, int(card), time_s)

    def merge(self, other: "KVTransferDomainDemands") -> None:
        for key, value in other.dma.items():
            self._merge_time(self.dma, key, value)
        for key, value in other.hbm_read.items():
            self._merge_time(self.hbm_read, key, value)
        for key, value in other.hbm_write.items():
            self._merge_time(self.hbm_write, key, value)
        for key, value in other.nvlink.items():
            self._merge_time(self.nvlink, key, value)
        for key, value in other.pcie.items():
            self._merge_time(self.pcie, key, value)


@dataclass(frozen=True)
class KVTransferOverlap:
    dma_overlapped_time_s: float = 0.0
    hbm_read_overlapped_time_s: float = 0.0
    hbm_write_overlapped_time_s: float = 0.0
    hbm_overlapped_time_s: float = 0.0
    nvlink_overlapped_time_s: float = 0.0
    pcie_overlapped_time_s: float = 0.0
    blocking_time_s: float = 0.0
    active_dma_domains: int = 0
    active_hbm_read_domains: int = 0
    active_hbm_write_domains: int = 0
    active_nvlink_domains: int = 0
    active_pcie_domains: int = 0


def reduce_transfer_demands(demands: KVTransferDomainDemands) -> KVTransferOverlap:
    dma_overlapped = max(demands.dma.values()) if demands.dma else 0.0
    hbm_read_overlapped = max(demands.hbm_read.values()) if demands.hbm_read else 0.0
    hbm_write_overlapped = max(demands.hbm_write.values()) if demands.hbm_write else 0.0
    nvlink_overlapped = max(demands.nvlink.values()) if demands.nvlink else 0.0
    pcie_overlapped = max(demands.pcie.values()) if demands.pcie else 0.0
    hbm_overlapped = hbm_read_overlapped + hbm_write_overlapped
    blocking_time_s = max(dma_overlapped, pcie_overlapped, hbm_overlapped + nvlink_overlapped)
    return KVTransferOverlap(
        dma_overlapped_time_s=dma_overlapped,
        hbm_read_overlapped_time_s=hbm_read_overlapped,
        hbm_write_overlapped_time_s=hbm_write_overlapped,
        hbm_overlapped_time_s=hbm_overlapped,
        nvlink_overlapped_time_s=nvlink_overlapped,
        pcie_overlapped_time_s=pcie_overlapped,
        blocking_time_s=blocking_time_s,
        active_dma_domains=len(demands.dma),
        active_hbm_read_domains=len(demands.hbm_read),
        active_hbm_write_domains=len(demands.hbm_write),
        active_nvlink_domains=len(demands.nvlink),
        active_pcie_domains=len(demands.pcie),
    )


@dataclass
class KVCacheStats:
    hits: int = 0
    l1_hits: int = 0
    l2_hits: int = 0
    l3_hits: int = 0
    misses: int = 0
    inserts: int = 0
    evictions: int = 0
    same_die_l1_to_l1: int = 0
    cross_die_l1_to_l1: int = 0
    l1_to_l2: int = 0
    same_die_l1_to_l2: int = 0
    cross_die_l1_to_l2: int = 0
    cross_card_l1_to_l2: int = 0
    l2_to_l3: int = 0
    cross_die_l2_to_l2: int = 0
    cross_card_l2_to_l2: int = 0
    l3_drops: int = 0
    callback_same_die: int = 0
    callback_cross_die: int = 0
    callback_cross_card: int = 0
    callback_l3_to_l1: int = 0


@dataclass
class KVAccessResult:
    is_hit: bool = False
    hit_tier: Optional[KVTier] = None
    inserted: bool = False
    same_die_l1_to_l1: int = 0
    cross_die_l1_to_l1: int = 0
    same_die_l1_to_l2: int = 0
    cross_die_l1_to_l2: int = 0
    cross_card_l1_to_l2: int = 0
    cross_die_l2_to_l2: int = 0
    cross_card_l2_to_l2: int = 0
    l2_to_l3: int = 0
    l3_drops: int = 0
    callback_same_die: int = 0
    callback_cross_die: int = 0
    callback_cross_card: int = 0
    callback_from_l3: int = 0
    migration_bytes: int = 0
    callback_time_s: float = 0.0
    eviction_time_s: float = 0.0
    same_die_time_s: float = 0.0
    cross_die_time_s: float = 0.0
    cross_card_time_s: float = 0.0
    pcie_time_s: float = 0.0
    callback_demands: KVTransferDomainDemands = field(default_factory=KVTransferDomainDemands)
    eviction_demands: KVTransferDomainDemands = field(default_factory=KVTransferDomainDemands)

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
    def l1_to_l2(self) -> int:
        return self.same_die_l1_to_l2 + self.cross_die_l1_to_l2 + self.cross_card_l1_to_l2

    @property
    def dma_blocks(self) -> int:
        return self.same_die_l1_to_l1 + self.same_die_l1_to_l2 + self.callback_same_die

    @property
    def pcie_blocks(self) -> int:
        return self.l2_to_l3 + self.callback_from_l3

    @property
    def migration_blocks(self) -> int:
        return int(self.migration_bytes > 0) if self.migration_bytes <= 0 else int(self.migration_bytes)


class KVCacheManager:
    """Topology-aware KV cache with L1 at (card, die, bank), L2 at (card, die), and global L3."""

    def __init__(
        self,
        kv_bytes_per_block: int,
        topology: dict,
        spare_ratio: float = 0.01,
    ):
        if kv_bytes_per_block <= 0:
            raise ValueError("kv_bytes_per_block must be > 0")
        if float(spare_ratio) < 0.0 or float(spare_ratio) >= 1.0:
            raise ValueError("spare_ratio must be in [0.0, 1.0)")

        self.kv_bytes_per_block = int(kv_bytes_per_block)
        self.spare_ratio = float(spare_ratio)
        self.num_cards = int(topology["num_cards"])
        self.num_die_packages_per_card = int(topology["num_die_packages_per_card"])
        self.banks_per_die = int(topology["banks_per_die"])
        self.l1_die_ids = tuple(int(v) for v in topology["l1_die_ids"])
        self.l2_die_ids = tuple(int(v) for v in topology["l2_die_ids"])
        self.l1_bank_capacity_bytes = int(topology["l1_bank_capacity_bytes"])
        self.l2_die_capacity_bytes = int(topology["l2_kv_die_capacity_bytes"])
        self.l3_capacity_bytes = int(topology["l3_kv_bytes"])
        self.card_total_bw_bps = float(topology["card_total_bw_bps"])
        self.nvlink_bw_bps = float(topology["nvlink_bw_bps"])
        self.dma_bw_bps = float(topology["dma_bw_bps"])
        self.pcie_bw_bps = float(topology["pcie_bw_bps"])

        self._l1_banks: Dict[Tuple[int, int, int], OrderedDict] = {}
        self._l1_used: Dict[Tuple[int, int, int], int] = {}
        for card in range(self.num_cards):
            for die in self.l1_die_ids:
                for bank in range(self.banks_per_die):
                    key = (card, die, bank)
                    self._l1_banks[key] = OrderedDict()
                    self._l1_used[key] = 0

        self._l2_dies: Dict[Tuple[int, int], OrderedDict] = {}
        self._l2_used: Dict[Tuple[int, int], int] = {}
        for card in range(self.num_cards):
            for die in self.l2_die_ids:
                key = (card, die)
                self._l2_dies[key] = OrderedDict()
                self._l2_used[key] = 0

        self._l3 = OrderedDict()
        self._l3_used = 0
        self._locations: Dict[int, KVLocation] = {}

        self.stats = KVCacheStats()
        self._l1_total_capacity_bytes = len(self._l1_banks) * self.l1_bank_capacity_bytes
        self._l2_total_capacity_bytes = len(self._l2_dies) * self.l2_die_capacity_bytes
        self.capacity_bytes = self._l1_total_capacity_bytes + self._l2_total_capacity_bytes + self.l3_capacity_bytes

    @property
    def used_bytes(self) -> int:
        return sum(self._l1_used.values()) + sum(self._l2_used.values()) + self._l3_used

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def tier_used_bytes(self, tier: KVTier) -> int:
        if tier == KVTier.L1:
            return sum(self._l1_used.values())
        if tier == KVTier.L2:
            return sum(self._l2_used.values())
        return self._l3_used

    def tier_capacity_bytes(self, tier: KVTier) -> int:
        if tier == KVTier.L1:
            return self._l1_total_capacity_bytes
        if tier == KVTier.L2:
            return self._l2_total_capacity_bytes
        return self.l3_capacity_bytes

    def _effective_capacity(self, capacity_bytes: int) -> int:
        reserve = int(capacity_bytes * self.spare_ratio)
        threshold = capacity_bytes - reserve
        if capacity_bytes >= self.kv_bytes_per_block:
            threshold = max(threshold, self.kv_bytes_per_block)
        return max(threshold, 0)

    def _l1_capacity_threshold(self) -> int:
        return self._effective_capacity(self.l1_bank_capacity_bytes)

    def _l2_capacity_threshold(self) -> int:
        return self._effective_capacity(self.l2_die_capacity_bytes)

    def _l3_capacity_threshold(self) -> int:
        return self._effective_capacity(self.l3_capacity_bytes)

    def assign_request_home(self, req_id: int) -> Tuple[int, int]:
        if not self.l1_die_ids:
            return 0, 0
        home_card = req_id % self.num_cards
        die_idx = (req_id // self.num_cards) % len(self.l1_die_ids)
        return home_card, self.l1_die_ids[die_idx]

    def bank_for_hash(self, hash_id: int) -> int:
        return int(hash_id % self.banks_per_die)

    def _valid_l1_bank_key(self, card: int, die: int, bank: int) -> bool:
        return (card, die, bank) in self._l1_banks

    def _valid_l2_die_key(self, card: int, die: int) -> bool:
        return (card, die) in self._l2_dies

    def locate(self, hash_id: int) -> Optional[KVLocation]:
        return self._locations.get(hash_id)

    def has(self, hash_id: int) -> bool:
        return hash_id in self._locations

    def _movement_time_s(self, src: KVLocation, dst: KVLocation) -> float:
        bytes_size = self.kv_bytes_per_block
        per_die_bw = (self.card_total_bw_bps / self.num_die_packages_per_card) if self.num_die_packages_per_card > 0 else 0.0

        if src.tier == KVTier.L3 or dst.tier == KVTier.L3:
            return (bytes_size / self.pcie_bw_bps) if self.pcie_bw_bps > 0 else 0.0
        if src.card == dst.card:
            if src.die == dst.die:
                return (2.0 * bytes_size / self.dma_bw_bps) if self.dma_bw_bps > 0 else 0.0
            return (2.0 * bytes_size / per_die_bw) if per_die_bw > 0 else 0.0
        if per_die_bw <= 0 or self.nvlink_bw_bps <= 0:
            return 0.0
        return (bytes_size / per_die_bw) + (bytes_size / self.nvlink_bw_bps) + (bytes_size / per_die_bw)

    def _record_transfer(self, result: KVAccessResult, src: KVLocation, dst: KVLocation, is_callback: bool) -> None:
        time_s = self._movement_time_s(src, dst)
        result.migration_bytes += self.kv_bytes_per_block
        if src.tier == KVTier.L3 or dst.tier == KVTier.L3:
            result.pcie_time_s += time_s
        elif src.card == dst.card and src.die == dst.die:
            result.same_die_time_s += time_s
        elif src.card == dst.card:
            result.cross_die_time_s += time_s
        else:
            result.cross_card_time_s += time_s

        if is_callback:
            result.callback_time_s += time_s
        else:
            result.eviction_time_s += time_s
        self._record_transfer_domains(result, src, dst, is_callback=is_callback)

    def _record_transfer_domains(self, result: KVAccessResult, src: KVLocation, dst: KVLocation, is_callback: bool) -> None:
        bytes_size = self.kv_bytes_per_block
        per_die_bw = (self.card_total_bw_bps / self.num_die_packages_per_card) if self.num_die_packages_per_card > 0 else 0.0
        pcie_time_s = (bytes_size / self.pcie_bw_bps) if self.pcie_bw_bps > 0 else 0.0
        hbm_stage_time_s = (bytes_size / per_die_bw) if per_die_bw > 0 else 0.0
        dma_time_s = (2.0 * bytes_size / self.dma_bw_bps) if self.dma_bw_bps > 0 else 0.0
        nvlink_time_s = (bytes_size / self.nvlink_bw_bps) if self.nvlink_bw_bps > 0 else 0.0
        demands = result.callback_demands if is_callback else result.eviction_demands

        if src.tier == KVTier.L3 or dst.tier == KVTier.L3:
            card = dst.card if src.tier == KVTier.L3 else src.card
            demands.add_pcie(card, pcie_time_s)
            return
        if src.card == dst.card:
            if src.die == dst.die:
                demands.add_dma(src.card, src.die, dma_time_s)
                return
            demands.add_hbm_read(src.card, src.die, hbm_stage_time_s)
            demands.add_hbm_write(dst.card, dst.die, hbm_stage_time_s)
            return
        demands.add_hbm_read(src.card, src.die, hbm_stage_time_s)
        demands.add_nvlink(src.card, dst.card, nvlink_time_s)
        demands.add_hbm_write(dst.card, dst.die, hbm_stage_time_s)

    def _record_move(self, src: KVLocation, dst: KVLocation, result: KVAccessResult, is_callback: bool = False) -> None:
        self._record_transfer(result, src, dst, is_callback=is_callback)

        if is_callback:
            if src.tier == KVTier.L3:
                self.stats.callback_l3_to_l1 += 1
                result.callback_from_l3 += 1
                return
            if src.card == dst.card and src.die == dst.die:
                self.stats.callback_same_die += 1
                result.callback_same_die += 1
            elif src.card == dst.card:
                self.stats.callback_cross_die += 1
                result.callback_cross_die += 1
            else:
                self.stats.callback_cross_card += 1
                result.callback_cross_card += 1
            return

        self.stats.evictions += 1
        if src.tier == KVTier.L1 and dst.tier == KVTier.L1:
            if src.card == dst.card and src.die == dst.die:
                self.stats.same_die_l1_to_l1 += 1
                result.same_die_l1_to_l1 += 1
            else:
                self.stats.cross_die_l1_to_l1 += 1
                result.cross_die_l1_to_l1 += 1
            return
        if src.tier == KVTier.L1 and dst.tier == KVTier.L2:
            self.stats.l1_to_l2 += 1
            if src.card == dst.card and src.die == dst.die:
                self.stats.same_die_l1_to_l2 += 1
                result.same_die_l1_to_l2 += 1
            elif src.card == dst.card:
                self.stats.cross_die_l1_to_l2 += 1
                result.cross_die_l1_to_l2 += 1
            else:
                self.stats.cross_card_l1_to_l2 += 1
                result.cross_card_l1_to_l2 += 1
            return
        if src.tier == KVTier.L2 and dst.tier == KVTier.L2:
            if src.card == dst.card:
                self.stats.cross_die_l2_to_l2 += 1
                result.cross_die_l2_to_l2 += 1
            else:
                self.stats.cross_card_l2_to_l2 += 1
                result.cross_card_l2_to_l2 += 1
            return
        if src.tier == KVTier.L2 and dst.tier == KVTier.L3:
            self.stats.l2_to_l3 += 1
            result.l2_to_l3 += 1

    def _remove(self, hash_id: int, location: KVLocation) -> None:
        if location.tier == KVTier.L1:
            key = (location.card, location.die, location.bank)
            if hash_id in self._l1_banks[key]:
                del self._l1_banks[key][hash_id]
                self._l1_used[key] -= self.kv_bytes_per_block
        elif location.tier == KVTier.L2:
            key = (location.card, location.die)
            if hash_id in self._l2_dies[key]:
                del self._l2_dies[key][hash_id]
                self._l2_used[key] -= self.kv_bytes_per_block
        else:
            if hash_id in self._l3:
                del self._l3[hash_id]
                self._l3_used -= self.kv_bytes_per_block
        self._locations.pop(hash_id, None)

    def _insert_l1_bank(self, hash_id: int, bank_key: Tuple[int, int, int]) -> None:
        self._l1_banks[bank_key][hash_id] = True
        self._l1_used[bank_key] += self.kv_bytes_per_block
        self._locations[hash_id] = KVLocation(KVTier.L1, bank_key[0], bank_key[1], bank_key[2])

    def _insert_l2_die(self, hash_id: int, die_key: Tuple[int, int]) -> None:
        self._l2_dies[die_key][hash_id] = True
        self._l2_used[die_key] += self.kv_bytes_per_block
        self._locations[hash_id] = KVLocation(KVTier.L2, die_key[0], die_key[1], None)

    def _insert_l3(self, hash_id: int) -> bool:
        cap = self._l3_capacity_threshold()
        if cap < self.kv_bytes_per_block:
            return False
        while self._l3_used + self.kv_bytes_per_block > cap:
            victim_hash, _ = self._l3.popitem(last=False)
            self._locations.pop(victim_hash, None)
            self._l3_used -= self.kv_bytes_per_block
            self.stats.evictions += 1
            self.stats.l3_drops += 1
        self._l3[hash_id] = True
        self._l3_used += self.kv_bytes_per_block
        self._locations[hash_id] = KVLocation(KVTier.L3, None, None, None)
        return True

    def _candidate_sort_key(self, source: KVLocation, candidate: KVLocation) -> Tuple[float, int, int, int, int]:
        score = self._movement_time_s(source, candidate) + self._movement_time_s(candidate, source)
        tier_priority = {KVTier.L1: 0, KVTier.L2: 1, KVTier.L3: 2}[candidate.tier]
        return (
            score,
            tier_priority,
            -1 if candidate.card is None else candidate.card,
            -1 if candidate.die is None else candidate.die,
            -1 if candidate.bank is None else candidate.bank,
        )

    def _l1_bank_has_space(self, bank_key: Tuple[int, int, int]) -> bool:
        return self._l1_used[bank_key] + self.kv_bytes_per_block <= self._l1_capacity_threshold()

    def _l2_die_has_space(self, die_key: Tuple[int, int]) -> bool:
        return self._l2_used[die_key] + self.kv_bytes_per_block <= self._l2_capacity_threshold()

    def _l1_candidates(
        self,
        card: int,
        die: int,
        include_same_die: bool,
        include_cross_die: bool,
        exclude_bank: Optional[Tuple[int, int, int]] = None,
    ) -> Iterable[KVLocation]:
        for bank_key in self._l1_banks.keys():
            bank_card, bank_die, bank_idx = bank_key
            if bank_card != card:
                continue
            if exclude_bank is not None and bank_key == exclude_bank:
                continue
            if bank_die == die and not include_same_die:
                continue
            if bank_die != die and not include_cross_die:
                continue
            if self._l1_bank_has_space(bank_key):
                yield KVLocation(KVTier.L1, bank_card, bank_die, bank_idx)

    def _same_card_l2_candidates(self, card: int, preferred_die: Optional[int] = None) -> Iterable[KVLocation]:
        ordered = []
        if preferred_die is not None and self._valid_l2_die_key(card, preferred_die):
            ordered.append((card, preferred_die))
        for die_key in self._l2_dies.keys():
            if die_key[0] != card:
                continue
            if preferred_die is not None and die_key[1] == preferred_die:
                continue
            ordered.append(die_key)
        for die_key in ordered:
            if self._l2_die_has_space(die_key):
                yield KVLocation(KVTier.L2, die_key[0], die_key[1], None)

    def _cross_card_l2_candidates(self, card: int) -> Iterable[KVLocation]:
        for die_key in self._l2_dies.keys():
            if die_key[0] == card:
                continue
            if self._l2_die_has_space(die_key):
                yield KVLocation(KVTier.L2, die_key[0], die_key[1], None)

    def _select_l1_victim_destination(self, source: KVLocation) -> KVLocation:
        candidates = []
        exclude_bank = (source.card, source.die, source.bank)
        candidates.extend(
            self._l1_candidates(
                source.card,
                source.die,
                include_same_die=True,
                include_cross_die=False,
                exclude_bank=exclude_bank,
            )
        )
        candidates.extend(self._same_card_l2_candidates(source.card, preferred_die=source.die))
        candidates.extend(
            self._l1_candidates(
                source.card,
                source.die,
                include_same_die=False,
                include_cross_die=True,
                exclude_bank=exclude_bank,
            )
        )
        candidates.extend(self._same_card_l2_candidates(source.card, preferred_die=None))
        candidates.extend(self._cross_card_l2_candidates(source.card))
        if self._l3_capacity_threshold() >= self.kv_bytes_per_block:
            candidates.append(KVLocation(KVTier.L3, None, None, None))
        if not candidates:
            return KVLocation(KVTier.L3, None, None, None)
        return min(candidates, key=lambda candidate: self._candidate_sort_key(source, candidate))

    def _select_l2_victim_destination(self, source: KVLocation) -> KVLocation:
        candidates = list(self._same_card_l2_candidates(source.card, preferred_die=None))
        candidates = [candidate for candidate in candidates if candidate.die != source.die]
        candidates.extend(self._cross_card_l2_candidates(source.card))
        if self._l3_capacity_threshold() >= self.kv_bytes_per_block:
            candidates.append(KVLocation(KVTier.L3, None, None, None))
        if not candidates:
            return KVLocation(KVTier.L3, None, None, None)
        return min(candidates, key=lambda candidate: self._candidate_sort_key(source, candidate))

    def _move_hash_to_candidate(self, hash_id: int, source: KVLocation, candidate: KVLocation, result: KVAccessResult) -> bool:
        if candidate.tier == KVTier.L1:
            bank_key = (candidate.card, candidate.die, candidate.bank)
            if not self._l1_bank_has_space(bank_key):
                return False
            self._insert_l1_bank(hash_id, bank_key)
        elif candidate.tier == KVTier.L2:
            die_key = (candidate.card, candidate.die)
            if not self._l2_die_has_space(die_key):
                return False
            self._insert_l2_die(hash_id, die_key)
        else:
            if not self._insert_l3(hash_id):
                self.stats.l3_drops += 1
                result.l3_drops += 1
                return False
        self._record_move(source, candidate, result, is_callback=False)
        return True

    def _evict_from_l1_bank(self, bank_key: Tuple[int, int, int], result: KVAccessResult) -> bool:
        entries = self._l1_banks[bank_key]
        if not entries:
            return False
        victim_hash, _ = entries.popitem(last=False)
        self._l1_used[bank_key] -= self.kv_bytes_per_block
        source = KVLocation(KVTier.L1, bank_key[0], bank_key[1], bank_key[2])
        self._locations.pop(victim_hash, None)
        candidate = self._select_l1_victim_destination(source)
        return self._move_hash_to_candidate(victim_hash, source, candidate, result)

    def _evict_from_l2_die(self, die_key: Tuple[int, int], result: KVAccessResult) -> bool:
        entries = self._l2_dies[die_key]
        if not entries:
            return False
        victim_hash, _ = entries.popitem(last=False)
        self._l2_used[die_key] -= self.kv_bytes_per_block
        source = KVLocation(KVTier.L2, die_key[0], die_key[1], None)
        self._locations.pop(victim_hash, None)
        candidate = self._select_l2_victim_destination(source)
        return self._move_hash_to_candidate(victim_hash, source, candidate, result)

    def _ensure_l1_bank_space(self, bank_key: Tuple[int, int, int], result: KVAccessResult) -> bool:
        cap = self._l1_capacity_threshold()
        if cap < self.kv_bytes_per_block:
            return False
        while self._l1_used[bank_key] + self.kv_bytes_per_block > cap:
            if not self._evict_from_l1_bank(bank_key, result):
                break
        return self._l1_used[bank_key] + self.kv_bytes_per_block <= cap

    def _ensure_l2_die_space(self, die_key: Tuple[int, int], result: KVAccessResult) -> bool:
        cap = self._l2_capacity_threshold()
        if cap < self.kv_bytes_per_block:
            return False
        while self._l2_used[die_key] + self.kv_bytes_per_block > cap:
            if not self._evict_from_l2_die(die_key, result):
                break
        return self._l2_used[die_key] + self.kv_bytes_per_block <= cap

    def _callback_to_l1(self, hash_id: int, source: KVLocation, target_card: int, target_die: int, result: KVAccessResult) -> bool:
        target_bank_key = (target_card, target_die, self.bank_for_hash(hash_id))
        if not self._valid_l1_bank_key(*target_bank_key):
            return False
        if not self._ensure_l1_bank_space(target_bank_key, result):
            return False
        self._insert_l1_bank(hash_id, target_bank_key)
        self._record_move(source, KVLocation(KVTier.L1, target_card, target_die, target_bank_key[2]), result, is_callback=True)
        return True

    def access(self, hash_id: int, target_card: Optional[int] = None, target_die: Optional[int] = None) -> KVAccessResult:
        result = KVAccessResult()
        location = self.locate(hash_id)

        if target_card is None or target_die is None:
            target_card, target_die = self.assign_request_home(hash_id)

        if location is None:
            self.stats.misses += 1
            target_bank_key = (target_card, target_die, self.bank_for_hash(hash_id))
            if self._valid_l1_bank_key(*target_bank_key) and self._ensure_l1_bank_space(target_bank_key, result):
                self._insert_l1_bank(hash_id, target_bank_key)
                self.stats.inserts += 1
                result.inserted = True
            return result

        if location.tier == KVTier.L1:
            self.stats.hits += 1
            self.stats.l1_hits += 1
            result.is_hit = True
            result.hit_tier = KVTier.L1
            bank_key = (location.card, location.die, location.bank)
            self._l1_banks[bank_key].move_to_end(hash_id)
            return result

        if location.tier == KVTier.L2:
            self.stats.hits += 1
            self.stats.l2_hits += 1
            result.is_hit = True
            result.hit_tier = KVTier.L2
            self._remove(hash_id, location)
            if not self._callback_to_l1(hash_id, location, target_card, target_die, result):
                self._insert_l2_die(hash_id, (location.card, location.die))
            return result

        self.stats.hits += 1
        self.stats.l3_hits += 1
        result.is_hit = True
        result.hit_tier = KVTier.L3
        self._remove(hash_id, location)
        if not self._callback_to_l1(hash_id, location, target_card, target_die, result):
            self._insert_l3(hash_id)
        return result

    def touch(self, hash_id: int) -> bool:
        location = self.locate(hash_id)
        if location is None:
            return False
        if location.tier == KVTier.L1:
            self._l1_banks[(location.card, location.die, location.bank)].move_to_end(hash_id)
        elif location.tier == KVTier.L2:
            self._l2_dies[(location.card, location.die)].move_to_end(hash_id)
        else:
            self._l3.move_to_end(hash_id)
        return True

    def probe(self, hash_id: int) -> bool:
        location = self.locate(hash_id)
        if location is None:
            self.stats.misses += 1
            return False
        self.stats.hits += 1
        if location.tier == KVTier.L1:
            self.stats.l1_hits += 1
        elif location.tier == KVTier.L2:
            self.stats.l2_hits += 1
        else:
            self.stats.l3_hits += 1
        self.touch(hash_id)
        return True

    def reserve_or_evict(self, required_bytes: int, tier: KVTier = KVTier.L1) -> int:
        if required_bytes <= 0:
            return 0
        cap = self.tier_capacity_bytes(tier)
        return -1 if required_bytes > cap else 0

    def insert(
        self,
        hash_id: int,
        tier: KVTier = KVTier.L1,
        target_card: int = 0,
        target_die: int = 0,
    ) -> bool:
        if self.has(hash_id):
            self.touch(hash_id)
            return True
        result = KVAccessResult()
        if tier == KVTier.L1:
            target_bank_key = (target_card, target_die, self.bank_for_hash(hash_id))
            if not self._valid_l1_bank_key(*target_bank_key):
                return False
            if not self._ensure_l1_bank_space(target_bank_key, result):
                return False
            self._insert_l1_bank(hash_id, target_bank_key)
            self.stats.inserts += 1
            return True
        if tier == KVTier.L2:
            die_key = (target_card, target_die)
            if not self._valid_l2_die_key(*die_key):
                return False
            if not self._ensure_l2_die_space(die_key, result):
                return False
            self._insert_l2_die(hash_id, die_key)
            self.stats.inserts += 1
            return True
        ok = self._insert_l3(hash_id)
        if ok:
            self.stats.inserts += 1
        return ok

    def snapshot(self) -> dict:
        l1_num_blocks = sum(len(entries) for entries in self._l1_banks.values())
        l2_num_blocks = sum(len(entries) for entries in self._l2_dies.values())
        l3_num_blocks = len(self._l3)
        return {
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "num_blocks": l1_num_blocks + l2_num_blocks + l3_num_blocks,
            "l1_num_blocks": l1_num_blocks,
            "l2_num_blocks": l2_num_blocks,
            "l3_num_blocks": l3_num_blocks,
            "l1_capacity_bytes": self._l1_total_capacity_bytes,
            "l2_capacity_bytes": self._l2_total_capacity_bytes,
            "l3_capacity_bytes": self.l3_capacity_bytes,
            "l1_effective_capacity_bytes": self._effective_capacity(self._l1_total_capacity_bytes),
            "l2_effective_capacity_bytes": self._effective_capacity(self._l2_total_capacity_bytes),
            "l3_effective_capacity_bytes": self._effective_capacity(self.l3_capacity_bytes),
            "spare_ratio": self.spare_ratio,
            "l1_used_bytes": sum(self._l1_used.values()),
            "l2_used_bytes": sum(self._l2_used.values()),
            "l3_used_bytes": self._l3_used,
            "hits": self.stats.hits,
            "l1_hits": self.stats.l1_hits,
            "l2_hits": self.stats.l2_hits,
            "l3_hits": self.stats.l3_hits,
            "misses": self.stats.misses,
            "inserts": self.stats.inserts,
            "evictions": self.stats.evictions,
            "same_die_l1_to_l1": self.stats.same_die_l1_to_l1,
            "cross_die_l1_to_l1": self.stats.cross_die_l1_to_l1,
            "l1_to_l2": self.stats.l1_to_l2,
            "same_die_l1_to_l2": self.stats.same_die_l1_to_l2,
            "cross_die_l1_to_l2": self.stats.cross_die_l1_to_l2,
            "cross_card_l1_to_l2": self.stats.cross_card_l1_to_l2,
            "cross_die_l2_to_l2": self.stats.cross_die_l2_to_l2,
            "cross_card_l2_to_l2": self.stats.cross_card_l2_to_l2,
            "l2_to_l3": self.stats.l2_to_l3,
            "l3_drops": self.stats.l3_drops,
            "callback_same_die": self.stats.callback_same_die,
            "callback_cross_die": self.stats.callback_cross_die,
            "callback_cross_card": self.stats.callback_cross_card,
            "callback_l3_to_l1": self.stats.callback_l3_to_l1,
            "banks_per_die": self.banks_per_die,
            "num_die_packages_per_card": self.num_die_packages_per_card,
            "l1_bank_capacity_bytes": self.l1_bank_capacity_bytes,
            "l2_die_capacity_bytes": self.l2_die_capacity_bytes,
        }
