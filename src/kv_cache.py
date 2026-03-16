from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Iterable, List, Optional, Set, Tuple


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
    same_chat_hits: int = 0
    cross_chat_hits: int = 0
    single_turn_hits: int = 0
    multi_turn_hits: int = 0
    replica_hits: int = 0
    replica_l1_hits: int = 0
    replica_l2_hits: int = 0
    replica_fanout_blocks: int = 0
    replica_evictions: int = 0
    broadcast_promotions: int = 0
    avoided_cross_card_callbacks: int = 0


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
    same_chat_hit: int = 0
    cross_chat_hit: int = 0
    single_turn_hit: int = 0
    multi_turn_hit: int = 0
    replica_hit: int = 0
    replica_l1_hit: int = 0
    replica_l2_hit: int = 0
    avoided_cross_card_callback: int = 0
    replica_fanout_blocks: int = 0
    replica_fanout_bytes: int = 0
    replica_fanout_time_s: float = 0.0

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


@dataclass
class KVCategoryStats:
    accesses: int = 0
    hits: int = 0
    reuse_gap_ema_s: float = 1.0


@dataclass
class KVRequestContext:
    chat_id: Optional[int] = None
    request_type: str = ""
    turn_class: str = "single"
    sim_time: float = 0.0
    target_card: Optional[int] = None
    target_die: Optional[int] = None


@dataclass
class KVBlockMetadata:
    owner_chat_id: Optional[int] = None
    request_type: str = ""
    turn_class: str = "single"
    insert_time_s: float = 0.0
    last_access_time_s: float = 0.0
    reuse_count: int = 0
    last_reuse_gap_s: float = 0.0
    predicted_ttl_s: float = 1.0
    last_home_card: Optional[int] = None
    last_home_die: Optional[int] = None
    distinct_cards_seen: Set[int] = field(default_factory=set)
    remote_hit_count: int = 0


class KVCacheManager:
    """Topology-aware KV cache with L1 at (card, die, bank), L2 at (card, die), and global L3."""

    def __init__(
        self,
        kv_bytes_per_block: int,
        topology: dict,
        policy: Optional[dict] = None,
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
        self.policy = dict(policy or {})
        self.eviction_policy = str(self.policy.get("EVICTION_POLICY", "lru")).lower()
        self.placement_policy = str(self.policy.get("PLACEMENT_POLICY", "all_unique")).lower()
        self.aware_policy = dict(self.policy.get("AWARE_POLICY", {}))
        self.replica_tier = str(self.policy.get("REPLICA_TIER", "AUTO")).upper()
        self.replica_reserve_ratio_l1 = float(self.policy.get("REPLICA_RESERVE_RATIO_L1", 0.0) or 0.0)
        self.replica_reserve_ratio_l2 = float(self.policy.get("REPLICA_RESERVE_RATIO_L2", 0.0) or 0.0)
        self._current_request_context = KVRequestContext()
        self._metadata: Dict[int, KVBlockMetadata] = {}
        self._category_stats: Dict[Tuple[str, str], KVCategoryStats] = {}
        self._global_reuse_gap_ema_s = 1.0

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
        self._replica_locations: Dict[int, Dict[int, KVLocation]] = {}
        self._replica_l1_cards: Dict[int, OrderedDict] = {}
        self._replica_l1_used: Dict[int, int] = {}
        self._replica_l2_cards: Dict[int, OrderedDict] = {}
        self._replica_l2_used: Dict[int, int] = {}
        for card in range(self.num_cards):
            self._replica_l1_cards[card] = OrderedDict()
            self._replica_l1_used[card] = 0
            self._replica_l2_cards[card] = OrderedDict()
            self._replica_l2_used[card] = 0

        self.stats = KVCacheStats()
        self._l1_total_capacity_bytes = len(self._l1_banks) * self.l1_bank_capacity_bytes
        self._l2_total_capacity_bytes = len(self._l2_dies) * self.l2_die_capacity_bytes
        self._l1_replica_capacity_per_card_bytes = int(
            (self._l1_total_capacity_bytes / max(self.num_cards, 1)) * self.replica_reserve_ratio_l1
        )
        self._l2_replica_capacity_per_card_bytes = int(
            (self._l2_total_capacity_bytes / max(self.num_cards, 1)) * self.replica_reserve_ratio_l2
        )
        self.capacity_bytes = self._l1_total_capacity_bytes + self._l2_total_capacity_bytes + self.l3_capacity_bytes

    @property
    def used_bytes(self) -> int:
        return (
            sum(self._l1_used.values())
            + sum(self._l2_used.values())
            + self._l3_used
            + sum(self._replica_l1_used.values())
            + sum(self._replica_l2_used.values())
        )

    @property
    def free_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def tier_used_bytes(self, tier: KVTier) -> int:
        if tier == KVTier.L1:
            return sum(self._l1_used.values()) + sum(self._replica_l1_used.values())
        if tier == KVTier.L2:
            return sum(self._l2_used.values()) + sum(self._replica_l2_used.values())
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
        canonical_cap = int(self.l1_bank_capacity_bytes * max(0.0, 1.0 - self.replica_reserve_ratio_l1))
        return self._effective_capacity(canonical_cap)

    def _l2_capacity_threshold(self) -> int:
        canonical_cap = int(self.l2_die_capacity_bytes * max(0.0, 1.0 - self.replica_reserve_ratio_l2))
        return self._effective_capacity(canonical_cap)

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

    def _category_key_from_meta(self, meta: KVBlockMetadata) -> Tuple[str, str]:
        return (str(meta.request_type or ""), str(meta.turn_class or "single"))

    def _category_key_from_context(self, context: KVRequestContext) -> Tuple[str, str]:
        return (str(context.request_type or ""), str(context.turn_class or "single"))

    def _category_stats_for(self, key: Tuple[str, str]) -> KVCategoryStats:
        if key not in self._category_stats:
            self._category_stats[key] = KVCategoryStats(reuse_gap_ema_s=self._global_reuse_gap_ema_s)
        return self._category_stats[key]

    def _movement_cost_reference_s(self) -> float:
        if self.kv_bytes_per_block <= 0:
            return 1.0
        return max(self.kv_bytes_per_block / max(self.pcie_bw_bps, 1.0), 1e-9)

    def _set_request_context(
        self,
        target_card: Optional[int],
        target_die: Optional[int],
        sim_time: float = 0.0,
        chat_id: Optional[int] = None,
        request_type: str = "",
        turn_class: str = "single",
    ) -> None:
        self._current_request_context = KVRequestContext(
            chat_id=chat_id,
            request_type=str(request_type or ""),
            turn_class=str(turn_class or "single"),
            sim_time=float(sim_time or 0.0),
            target_card=target_card,
            target_die=target_die,
        )

    def _bootstrap_ttl_s(self, request_type: str, turn_class: str) -> float:
        stats = self._category_stats.get((str(request_type or ""), str(turn_class or "single")))
        base = stats.reuse_gap_ema_s if stats is not None else self._global_reuse_gap_ema_s
        return max(base * float(self.aware_policy.get("TTL_SAFETY_FACTOR", 1.0)), 1e-6)

    def _ensure_metadata(self, hash_id: int) -> KVBlockMetadata:
        if hash_id not in self._metadata:
            ctx = self._current_request_context
            meta = KVBlockMetadata(
                owner_chat_id=ctx.chat_id,
                request_type=ctx.request_type,
                turn_class=ctx.turn_class,
                insert_time_s=ctx.sim_time,
                last_access_time_s=ctx.sim_time,
                predicted_ttl_s=self._bootstrap_ttl_s(ctx.request_type, ctx.turn_class),
                last_home_card=ctx.target_card,
                last_home_die=ctx.target_die,
            )
            if ctx.target_card is not None:
                meta.distinct_cards_seen.add(int(ctx.target_card))
            self._metadata[hash_id] = meta
        return self._metadata[hash_id]

    def _score_block(self, hash_id: int, source: Optional[KVLocation] = None) -> float:
        if self.eviction_policy != "aware":
            return 0.0
        meta = self._metadata.get(hash_id)
        if meta is None:
            return -1.0
        ctx = self._current_request_context
        age_s = max(0.0, float(ctx.sim_time) - float(meta.last_access_time_s))
        ttl_s = max(float(meta.predicted_ttl_s), 1e-6)
        ttl_survival = math.exp(-age_s / ttl_s)
        hotness_cap = max(float(self.aware_policy.get("HOTNESS_CAP", 8)), 1.0)
        hotness = min(float(meta.reuse_count), hotness_cap) / hotness_cap
        distinct_cards = min(len(meta.distinct_cards_seen), self.num_cards) / max(self.num_cards, 1)
        locality = 1.0 if ctx.chat_id is not None and meta.owner_chat_id == ctx.chat_id else 0.0
        single_turn = 1.0 if meta.turn_class == "single" and bool(self.aware_policy.get("SINGLE_TURN_BIAS", True)) else 0.0
        expected_cost = 0.0
        if source is not None and ctx.target_card is not None and ctx.target_die is not None:
            expected_cost = self._movement_time_s(
                source,
                KVLocation(KVTier.L1, ctx.target_card, ctx.target_die, self.bank_for_hash(hash_id)),
            ) / self._movement_cost_reference_s()
        return (
            float(self.aware_policy.get("RECENCY_WEIGHT", 1.0)) * ttl_survival
            + float(self.aware_policy.get("REUSE_WEIGHT", 1.0)) * (0.5 * hotness + 0.5 * distinct_cards)
            + float(self.aware_policy.get("LOCALITY_WEIGHT", 1.0)) * (locality + 0.5 * single_turn)
            + float(self.aware_policy.get("COST_WEIGHT", 1.0)) * expected_cost
        )

    def _touch_metadata(self, hash_id: int, source: Optional[KVLocation], is_hit: bool) -> None:
        ctx = self._current_request_context
        meta = self._ensure_metadata(hash_id)
        age_s = max(0.0, float(ctx.sim_time) - float(meta.last_access_time_s))
        category_stats = self._category_stats_for(self._category_key_from_context(ctx))
        category_stats.accesses += 1
        if is_hit:
            category_stats.hits += 1
        if is_hit and age_s > 0.0:
            alpha = 0.2
            category_stats.reuse_gap_ema_s = (
                (1.0 - alpha) * category_stats.reuse_gap_ema_s + alpha * age_s
            )
            self._global_reuse_gap_ema_s = (
                (1.0 - alpha) * self._global_reuse_gap_ema_s + alpha * age_s
            )
            meta.last_reuse_gap_s = age_s
            meta.predicted_ttl_s = max(
                category_stats.reuse_gap_ema_s * float(self.aware_policy.get("TTL_SAFETY_FACTOR", 1.0)),
                1e-6,
            )
            meta.reuse_count += 1
        elif meta.predicted_ttl_s <= 0.0:
            meta.predicted_ttl_s = self._bootstrap_ttl_s(ctx.request_type, ctx.turn_class)
        meta.owner_chat_id = ctx.chat_id
        meta.request_type = ctx.request_type
        meta.turn_class = ctx.turn_class
        meta.last_access_time_s = float(ctx.sim_time)
        meta.last_home_card = ctx.target_card
        meta.last_home_die = ctx.target_die
        if ctx.target_card is not None:
            meta.distinct_cards_seen.add(int(ctx.target_card))
        if is_hit and source is not None and ctx.target_card is not None and (
            source.card is None or source.card != ctx.target_card
        ):
            meta.remote_hit_count += 1

    def _record_hit_classification(self, result: KVAccessResult, hash_id: int) -> None:
        meta = self._ensure_metadata(hash_id)
        ctx = self._current_request_context
        if ctx.chat_id is not None and meta.owner_chat_id == ctx.chat_id:
            self.stats.same_chat_hits += 1
            result.same_chat_hit = 1
        else:
            self.stats.cross_chat_hits += 1
            result.cross_chat_hit = 1
        if meta.turn_class == "single":
            self.stats.single_turn_hits += 1
            result.single_turn_hit = 1
        else:
            self.stats.multi_turn_hits += 1
            result.multi_turn_hit = 1

    def _replica_location_for_card(self, hash_id: int, card: int, tier_name: str, preferred_die: Optional[int] = None) -> Optional[KVLocation]:
        tier_name = str(tier_name or "AUTO").upper()
        if tier_name == "L1":
            if not self.l1_die_ids:
                return None
            die = preferred_die if preferred_die in self.l1_die_ids else self.l1_die_ids[0]
            return KVLocation(KVTier.L1, card, die, self.bank_for_hash(hash_id))
        if tier_name == "L2":
            if not self.l2_die_ids:
                return None
            die = preferred_die if preferred_die in self.l2_die_ids else self.l2_die_ids[0]
            return KVLocation(KVTier.L2, card, die, None)
        return None

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
            victim_hash = self._select_victim_hash(
                self._l3,
                lambda h: KVLocation(KVTier.L3, None, None, None),
            )
            if victim_hash is None:
                break
            self._l3.pop(victim_hash, None)
            self._locations.pop(victim_hash, None)
            self._l3_used -= self.kv_bytes_per_block
            self.stats.evictions += 1
            self.stats.l3_drops += 1
        if self._l3_used + self.kv_bytes_per_block > cap:
            return False
        self._l3[hash_id] = True
        self._l3_used += self.kv_bytes_per_block
        self._locations[hash_id] = KVLocation(KVTier.L3, None, None, None)
        return True

    def _select_victim_hash(self, entries: OrderedDict, location_factory) -> Optional[int]:
        if not entries:
            return None
        if self.eviction_policy != "aware":
            victim_hash = next(iter(entries.keys()))
            return victim_hash
        victim_hash = None
        victim_score = None
        for hash_id in entries.keys():
            score = self._score_block(hash_id, location_factory(hash_id))
            if victim_score is None or score < victim_score:
                victim_hash = hash_id
                victim_score = score
        return victim_hash

    def _replica_capacity_threshold(self, tier: KVTier) -> int:
        if tier == KVTier.L1:
            return self._effective_capacity(self._l1_replica_capacity_per_card_bytes)
        if tier == KVTier.L2:
            return self._effective_capacity(self._l2_replica_capacity_per_card_bytes)
        return 0

    def _replica_pool_for(self, tier: KVTier, card: int):
        if tier == KVTier.L1:
            return self._replica_l1_cards[card], self._replica_l1_used, self._l1_replica_capacity_per_card_bytes
        return self._replica_l2_cards[card], self._replica_l2_used, self._l2_replica_capacity_per_card_bytes

    def _replica_has_local(self, hash_id: int, card: Optional[int]) -> Optional[KVLocation]:
        if card is None:
            return None
        location = self._replica_locations.get(hash_id, {}).get(int(card))
        return location

    def _remove_replica(self, hash_id: int, card: int) -> None:
        location = self._replica_locations.get(hash_id, {}).pop(card, None)
        if location is None:
            return
        if location.tier == KVTier.L1:
            self._replica_l1_cards[card].pop(hash_id, None)
            self._replica_l1_used[card] = max(0, self._replica_l1_used[card] - self.kv_bytes_per_block)
        elif location.tier == KVTier.L2:
            self._replica_l2_cards[card].pop(hash_id, None)
            self._replica_l2_used[card] = max(0, self._replica_l2_used[card] - self.kv_bytes_per_block)
        if not self._replica_locations.get(hash_id):
            self._replica_locations.pop(hash_id, None)

    def _ensure_replica_space(self, tier: KVTier, card: int) -> bool:
        entries, used_map, _ = self._replica_pool_for(tier, card)
        cap = self._replica_capacity_threshold(tier)
        if cap < self.kv_bytes_per_block:
            return False
        while used_map[card] + self.kv_bytes_per_block > cap:
            victim_hash = self._select_victim_hash(entries, lambda h: self._replica_locations.get(h, {}).get(card))
            if victim_hash is None:
                break
            self._remove_replica(victim_hash, card)
            self.stats.replica_evictions += 1
        return used_map[card] + self.kv_bytes_per_block <= cap

    def _record_replica_fanout(self, result: KVAccessResult, src: KVLocation, dst: KVLocation) -> None:
        time_s = self._movement_time_s(src, dst)
        result.replica_fanout_blocks += 1
        result.replica_fanout_bytes += self.kv_bytes_per_block
        result.replica_fanout_time_s += time_s
        self.stats.replica_fanout_blocks += 1

    def _insert_replica(self, hash_id: int, location: KVLocation) -> bool:
        if location.card is None:
            return False
        tier = location.tier
        card = int(location.card)
        if self._replica_locations.get(hash_id, {}).get(card) is not None:
            return True
        if not self._ensure_replica_space(tier, card):
            return False
        if tier == KVTier.L1:
            self._replica_l1_cards[card][hash_id] = location
            self._replica_l1_used[card] += self.kv_bytes_per_block
        else:
            self._replica_l2_cards[card][hash_id] = location
            self._replica_l2_used[card] += self.kv_bytes_per_block
        self._replica_locations.setdefault(hash_id, {})[card] = location
        return True

    def _record_replica_hit(self, hash_id: int, location: KVLocation, result: KVAccessResult) -> None:
        self.stats.hits += 1
        self.stats.replica_hits += 1
        result.is_hit = True
        result.hit_tier = location.tier
        result.replica_hit = 1
        self._record_hit_classification(result, hash_id)
        if location.tier == KVTier.L1:
            self.stats.l1_hits += 1
            self.stats.replica_l1_hits += 1
            result.replica_l1_hit = 1
            self._replica_l1_cards[int(location.card)].move_to_end(hash_id)
        else:
            self.stats.l2_hits += 1
            self.stats.replica_l2_hits += 1
            result.replica_l2_hit = 1
            self._replica_l2_cards[int(location.card)].move_to_end(hash_id)
            target = KVLocation(KVTier.L1, self._current_request_context.target_card, self._current_request_context.target_die, self.bank_for_hash(hash_id))
            self._record_transfer(result, location, target, is_callback=True)
            if location.card is not None and self._current_request_context.target_card is not None and location.card != self._current_request_context.target_card:
                self.stats.avoided_cross_card_callbacks += 1
                result.avoided_cross_card_callback = 1
        self._touch_metadata(hash_id, location, is_hit=True)

    def _maybe_broadcast_replicas(self, hash_id: int, canonical_location: Optional[KVLocation], result: KVAccessResult) -> None:
        if self.placement_policy != "hotset_broadcast":
            return
        meta = self._metadata.get(hash_id)
        if meta is None:
            return
        min_cards = int(self.aware_policy.get("REPLICA_MIN_DISTINCT_CARDS", 3))
        min_remote_hits = int(self.aware_policy.get("REPLICA_MIN_REMOTE_HITS", 8))
        score_threshold = float(self.aware_policy.get("REPLICA_SCORE_THRESHOLD", 0.0))
        score = self._score_block(hash_id, canonical_location)
        if len(meta.distinct_cards_seen) < min_cards or meta.remote_hit_count < min_remote_hits or score < score_threshold:
            return
        tier_name = self.replica_tier
        if tier_name == "AUTO":
            tier_name = "L2" if self._l2_replica_capacity_per_card_bytes >= self.kv_bytes_per_block and self.l2_die_ids else "L1"
        replica_tier = KVTier.L2 if tier_name == "L2" else KVTier.L1
        promoted = False
        for card in range(self.num_cards):
            if canonical_location is not None and canonical_location.card == card:
                continue
            if self._replica_locations.get(hash_id, {}).get(card) is not None:
                continue
            location = self._replica_location_for_card(hash_id, card, tier_name, preferred_die=self._current_request_context.target_die)
            if location is None:
                continue
            if self._insert_replica(hash_id, location):
                promoted = True
                if canonical_location is not None:
                    self._record_replica_fanout(result, canonical_location, location)
        if promoted:
            self.stats.broadcast_promotions += 1

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
        victim_hash = self._select_victim_hash(
            entries,
            lambda h: KVLocation(KVTier.L1, bank_key[0], bank_key[1], bank_key[2]),
        )
        if victim_hash is None:
            return False
        entries.pop(victim_hash, None)
        self._l1_used[bank_key] -= self.kv_bytes_per_block
        source = KVLocation(KVTier.L1, bank_key[0], bank_key[1], bank_key[2])
        self._locations.pop(victim_hash, None)
        candidate = self._select_l1_victim_destination(source)
        return self._move_hash_to_candidate(victim_hash, source, candidate, result)

    def _evict_from_l2_die(self, die_key: Tuple[int, int], result: KVAccessResult) -> bool:
        entries = self._l2_dies[die_key]
        if not entries:
            return False
        victim_hash = self._select_victim_hash(
            entries,
            lambda h: KVLocation(KVTier.L2, die_key[0], die_key[1], None),
        )
        if victim_hash is None:
            return False
        entries.pop(victim_hash, None)
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

    def access(
        self,
        hash_id: int,
        target_card: Optional[int] = None,
        target_die: Optional[int] = None,
        sim_time: float = 0.0,
        chat_id: Optional[int] = None,
        request_type: str = "",
        turn_class: str = "single",
    ) -> KVAccessResult:
        result = KVAccessResult()
        location = self.locate(hash_id)

        if target_card is None or target_die is None:
            target_card, target_die = self.assign_request_home(hash_id)
        self._set_request_context(
            target_card=target_card,
            target_die=target_die,
            sim_time=sim_time,
            chat_id=chat_id,
            request_type=request_type,
            turn_class=turn_class,
        )

        local_replica = self._replica_has_local(hash_id, target_card)
        canonical_local_l1 = (
            location is not None
            and location.tier == KVTier.L1
            and location.card == target_card
            and location.die == target_die
        )

        if local_replica is not None and not canonical_local_l1:
            self._record_replica_hit(hash_id, local_replica, result)
            if location is not None and location.card is not None and target_card is not None and location.card != target_card:
                self.stats.avoided_cross_card_callbacks += 1
                result.avoided_cross_card_callback = 1
            return result

        if location is None:
            self.stats.misses += 1
            target_bank_key = (target_card, target_die, self.bank_for_hash(hash_id))
            inserted = False
            if self._valid_l1_bank_key(*target_bank_key) and self._ensure_l1_bank_space(target_bank_key, result):
                self._insert_l1_bank(hash_id, target_bank_key)
                self.stats.inserts += 1
                result.inserted = True
                inserted = True
            elif self._valid_l2_die_key(target_card, target_die) and self._ensure_l2_die_space((target_card, target_die), result):
                self._insert_l2_die(hash_id, (target_card, target_die))
                self.stats.inserts += 1
                result.inserted = True
                inserted = True
            elif self._insert_l3(hash_id):
                self.stats.inserts += 1
                result.inserted = True
                inserted = True
            self._touch_metadata(hash_id, None, is_hit=False)
            if inserted:
                self._maybe_broadcast_replicas(hash_id, self.locate(hash_id), result)
            return result

        if location.tier == KVTier.L1:
            self.stats.hits += 1
            self.stats.l1_hits += 1
            result.is_hit = True
            result.hit_tier = KVTier.L1
            self._record_hit_classification(result, hash_id)
            bank_key = (location.card, location.die, location.bank)
            if location.card == target_card and location.die == target_die:
                self._l1_banks[bank_key].move_to_end(hash_id)
            else:
                self._remove(hash_id, location)
                if not self._callback_to_l1(hash_id, location, target_card, target_die, result):
                    self._insert_l1_bank(hash_id, bank_key)
            self._touch_metadata(hash_id, location, is_hit=True)
            self._maybe_broadcast_replicas(hash_id, self.locate(hash_id), result)
            return result

        if location.tier == KVTier.L2:
            self.stats.hits += 1
            self.stats.l2_hits += 1
            result.is_hit = True
            result.hit_tier = KVTier.L2
            self._record_hit_classification(result, hash_id)
            self._remove(hash_id, location)
            if not self._callback_to_l1(hash_id, location, target_card, target_die, result):
                self._insert_l2_die(hash_id, (location.card, location.die))
            self._touch_metadata(hash_id, location, is_hit=True)
            self._maybe_broadcast_replicas(hash_id, self.locate(hash_id), result)
            return result

        self.stats.hits += 1
        self.stats.l3_hits += 1
        result.is_hit = True
        result.hit_tier = KVTier.L3
        self._record_hit_classification(result, hash_id)
        self._remove(hash_id, location)
        if not self._callback_to_l1(hash_id, location, target_card, target_die, result):
            self._insert_l3(hash_id)
        self._touch_metadata(hash_id, location, is_hit=True)
        self._maybe_broadcast_replicas(hash_id, self.locate(hash_id), result)
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
        replica_l1_num_blocks = sum(len(entries) for entries in self._replica_l1_cards.values())
        replica_l2_num_blocks = sum(len(entries) for entries in self._replica_l2_cards.values())
        return {
            "capacity_bytes": self.capacity_bytes,
            "used_bytes": self.used_bytes,
            "free_bytes": self.free_bytes,
            "num_blocks": l1_num_blocks + l2_num_blocks + l3_num_blocks,
            "l1_num_blocks": l1_num_blocks,
            "l2_num_blocks": l2_num_blocks,
            "l3_num_blocks": l3_num_blocks,
            "replica_l1_num_blocks": replica_l1_num_blocks,
            "replica_l2_num_blocks": replica_l2_num_blocks,
            "l1_capacity_bytes": self._l1_total_capacity_bytes,
            "l2_capacity_bytes": self._l2_total_capacity_bytes,
            "l3_capacity_bytes": self.l3_capacity_bytes,
            "l1_effective_capacity_bytes": self._effective_capacity(self._l1_total_capacity_bytes),
            "l2_effective_capacity_bytes": self._effective_capacity(self._l2_total_capacity_bytes),
            "l3_effective_capacity_bytes": self._effective_capacity(self.l3_capacity_bytes),
            "replica_l1_capacity_per_card_bytes": self._l1_replica_capacity_per_card_bytes,
            "replica_l2_capacity_per_card_bytes": self._l2_replica_capacity_per_card_bytes,
            "spare_ratio": self.spare_ratio,
            "l1_used_bytes": sum(self._l1_used.values()),
            "l2_used_bytes": sum(self._l2_used.values()),
            "l3_used_bytes": self._l3_used,
            "replica_l1_used_bytes": sum(self._replica_l1_used.values()),
            "replica_l2_used_bytes": sum(self._replica_l2_used.values()),
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
            "same_chat_hits": self.stats.same_chat_hits,
            "cross_chat_hits": self.stats.cross_chat_hits,
            "single_turn_hits": self.stats.single_turn_hits,
            "multi_turn_hits": self.stats.multi_turn_hits,
            "replica_hits": self.stats.replica_hits,
            "replica_l1_hits": self.stats.replica_l1_hits,
            "replica_l2_hits": self.stats.replica_l2_hits,
            "replica_fanout_blocks": self.stats.replica_fanout_blocks,
            "replica_evictions": self.stats.replica_evictions,
            "broadcast_promotions": self.stats.broadcast_promotions,
            "avoided_cross_card_callbacks": self.stats.avoided_cross_card_callbacks,
            "eviction_policy": self.eviction_policy,
            "placement_policy": self.placement_policy,
            "replica_tier": self.replica_tier,
            "replica_reserve_ratio_l1": self.replica_reserve_ratio_l1,
            "replica_reserve_ratio_l2": self.replica_reserve_ratio_l2,
            "banks_per_die": self.banks_per_die,
            "num_die_packages_per_card": self.num_die_packages_per_card,
            "l1_bank_capacity_bytes": self.l1_bank_capacity_bytes,
            "l2_die_capacity_bytes": self.l2_die_capacity_bytes,
        }
