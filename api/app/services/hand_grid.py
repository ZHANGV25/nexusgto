"""Hand-grid and card helpers shared by the game model and the search wrapper.

The 169-class HAND_GRID, card <-> int encoding (rank<<2 | suit, A=12),
per-class combo expansion against blocked cards, and off-tree bet-size
snapping.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

# 169 hand classes in the same order as the frontend grid.
# Row 0-12 = A,K,Q,J,T,9,8,7,6,5,4,3,2
# col < row → suited (upper triangle), col > row → offsuit (lower triangle)
RANKS = "AKQJT98765432"

def generate_hand_grid() -> List[str]:
    """Generate 169 hand names in the exact order the frontend expects."""
    hands: List[str] = []
    for row in range(13):
        for col in range(13):
            r1 = RANKS[row]
            r2 = RANKS[col]
            if row == col:
                hands.append(f"{r1}{r2}")
            elif col < row:
                hands.append(f"{r2}{r1}s")
            else:
                hands.append(f"{r1}{r2}o")
    return hands


HAND_GRID: List[str] = generate_hand_grid()


# Action labels follow the solver's tree vocabulary ("Bet 75%", "All-in").
def _bet_size_label(frac: float) -> str:
    """Convert a pot-fraction bet size to its solver label.

    e.g. 0.33 -> "Bet 33%", 0.75 -> "Bet 75%", 1.5 -> "Bet 150%".
    """
    return f"Bet {int(round(frac * 100))}%"


# Tolerance below which an off-tree amount is "close enough" to snap to the
# nearest tree size. A bet within 15% of a tree size is snapped; otherwise
# the exact size is added to the tree vocabulary for a nested re-solve
# (Pluribus Algorithm 2, lines 4-7).
_OFF_TREE_SNAP_TOLERANCE = 0.15


def _fraction_is_on_tree(frac: float, tree_sizes: Tuple[float, ...]) -> bool:
    """Check if a pot-fraction is close enough to a tree size to snap."""
    if frac >= 5.0:
        return True  # counts as all-in
    for s in tree_sizes:
        rel = abs(frac - s) / max(s, 1e-6)
        if rel <= _OFF_TREE_SNAP_TOLERANCE:
            return True
    return False


def _snap_amount_to_tree(
    amount: float,
    pot: float,
    tree_sizes: Tuple[float, ...],
) -> str:
    """Snap an off-tree raise amount to the nearest tree action label.

    Called after nested re-solve (if applicable) has already added any
    significantly off-tree sizes to the tree vocabulary. At that point,
    the `amount` should map exactly or near-exactly to one of tree_sizes.

    `tree_sizes` is required (no default) — the runtime tree's first-
    raise vocab comes from the loaded blueprint's metadata.
    """
    if pot <= 0:
        # Fall back to the smallest tree size when there is no pot signal.
        # Picking the smallest minimizes EV damage if the snap is wrong.
        smallest = min(tree_sizes)
        return _bet_size_label(smallest)
    frac = amount / pot
    # All-in detection: any bet >= 5x pot is treated as all-in.
    if frac >= 5.0:
        return "All-in"
    # Find nearest of tree_sizes
    closest_frac = min(tree_sizes, key=lambda f: abs(f - frac))
    return _bet_size_label(closest_frac)


_SUIT_ORDER: Tuple[str, ...] = ("c", "d", "h", "s")
_RANK_TO_INT: Dict[str, int] = {r: 12 - i for i, r in enumerate(RANKS)}


def _card_to_int(card: str) -> int:
    """Card string -> int (0-51), matching solver/python/engine/cards.py:card_to_int."""
    rank_char, suit_char = card[0], card[1]
    rank = _RANK_TO_INT[rank_char.upper()]
    suit = _SUIT_ORDER.index(suit_char.lower())
    return rank * 4 + suit


def _int_to_card(c: int) -> str:
    rank_char = RANKS[12 - (c >> 2)]
    suit_char = _SUIT_ORDER[c & 3]
    return rank_char + suit_char


GridK = Union[int, str]
HandGridMetadata = Tuple[str, int, int]


def _grid_k_is_iso(k: GridK) -> bool:
    """``ISO``: one row per board-relative suit-isomorphism class (see
    app.services.suit_isomorphism)."""
    return isinstance(k, str) and k.strip().upper() == "ISO"


def _grid_k_limit(k: GridK) -> Optional[int]:
    """Normalize a grid dose. ``None`` means all live combos (ALL, and ISO
    before its grouping step)."""
    if isinstance(k, str):
        if k.strip().upper() in ("ALL", "ISO"):
            return None
        raise ValueError(
            f"grid K must be a positive integer, ALL or ISO, got {k!r}")
    if isinstance(k, bool) or int(k) != k or int(k) <= 0:
        raise ValueError(f"grid K must be a positive integer or ALL, got {k!r}")
    return int(k)


def _hand_class_combos(
    hand: str, blocked: set, k: GridK = 1,
) -> List[Tuple[int, int]]:
    """Return the first up-to-K live combos for one 169 hand class.

    The complete live list is generated in the original representative-
    picker order. Every finite K is therefore a prefix of every larger K and
    ``ALL``. No blocked or duplicate combo is emitted.
    """
    combos: List[Tuple[int, int]] = []
    if len(hand) == 2:
        rank = _RANK_TO_INT[hand[0]]
        for i, _s0 in enumerate(_SUIT_ORDER):
            c0 = rank * 4 + i
            if c0 in blocked:
                continue
            for j, _s1 in enumerate(_SUIT_ORDER):
                if j <= i:
                    continue
                c1 = rank * 4 + j
                if c1 not in blocked:
                    combos.append((c0, c1))
    else:
        r0 = _RANK_TO_INT[hand[0]]
        r1 = _RANK_TO_INT[hand[1]]
        if hand.endswith("s"):
            for i, _suit in enumerate(_SUIT_ORDER):
                c0 = r0 * 4 + i
                c1 = r1 * 4 + i
                if c0 not in blocked and c1 not in blocked:
                    combos.append((c0, c1))
        else:
            for i, _s0 in enumerate(_SUIT_ORDER):
                c0 = r0 * 4 + i
                if c0 in blocked:
                    continue
                for j, _s1 in enumerate(_SUIT_ORDER):
                    if i == j:
                        continue
                    c1 = r1 * 4 + j
                    if c1 not in blocked:
                        combos.append((c0, c1))

    limit = _grid_k_limit(k)
    return combos if limit is None else combos[:limit]


def _build_hand_grid_range_with_multiplicity(
    blocked: set,
    weights: Optional[Dict[str, float]] = None,
    k: GridK = 1,
    board: Optional[Sequence[int]] = None,
) -> Tuple[
    List[Tuple[int, int, float, str]], List[HandGridMetadata], List[int]
]:
    """Rows, aligned ``(class_name, combo_index, n_live)`` metadata, and the
    per-row multiplicity (how many live physical combos the row stands for).

    K=1 / integer K / ALL emit physical combos (multiplicity 1 each). ISO
    groups each class's live combos by board-relative suit-isomorphism
    (``suit_isomorphism.canonical_combo``), emits one row per group holding
    the canonical combo with weight ``class_weight * len(group) / n_live``,
    so class weight is conserved and each row carries its multiplicity; it
    needs ``board`` for the flush-relevant suits. Row order: HAND_GRID class
    order, then canonical combo ascending. A fully blocked class keeps its
    weight-zero placeholder (multiplicity 0).
    """
    iso = _grid_k_is_iso(k)
    if iso and board is None:
        raise ValueError("grid K=ISO needs the board to pool suits")
    rows: List[Tuple[int, int, float, str]] = []
    metadata: List[HandGridMetadata] = []
    multiplicity: List[int] = []
    limit = None if iso else _grid_k_limit(k)
    for hand in HAND_GRID:
        live_combos = _hand_class_combos(hand, blocked, k="ALL")
        n_live = len(live_combos)
        if not live_combos:
            rows.append((0, 1, 0.0, hand))
            metadata.append((hand, -1, 0))
            multiplicity.append(0)
            continue
        class_weight = weights.get(hand, 1.0) if weights is not None else 1.0
        if iso:
            from app.services.suit_isomorphism import group_by_canonical
            groups = group_by_canonical(live_combos, board or ())
            for combo_index, (canon, members) in enumerate(groups):
                rows.append((
                    canon[0], canon[1],
                    class_weight * len(members) / n_live, hand,
                ))
                metadata.append((hand, combo_index, n_live))
                multiplicity.append(len(members))
            continue
        emitted = live_combos if limit is None else live_combos[:limit]
        row_weight = class_weight / len(emitted)
        for combo_index, (c0, c1) in enumerate(emitted):
            rows.append((c0, c1, row_weight, hand))
            metadata.append((hand, combo_index, n_live))
            multiplicity.append(1)
    return rows, metadata, multiplicity


def _build_hand_grid_range_with_metadata(
    blocked: set,
    weights: Optional[Dict[str, float]] = None,
    k: GridK = 1,
    board: Optional[Sequence[int]] = None,
) -> Tuple[List[Tuple[int, int, float, str]], List[HandGridMetadata]]:
    """Build rows plus aligned ``(class_name, combo_index, n_live)`` data."""
    rows, metadata, _multiplicity = _build_hand_grid_range_with_multiplicity(
        blocked, weights=weights, k=k, board=board,
    )
    return rows, metadata


def _build_hand_grid_range(
    blocked: set,
    weights: Optional[Dict[str, float]] = None,
    k: GridK = 1,
    board: Optional[Sequence[int]] = None,
) -> List[Tuple[int, int, float, str]]:
    """Build a K-combo-per-class range in ``HAND_GRID`` order.

    Args:
        blocked: Set of card ints (0-51) that are unavailable (board cards).
        weights: Optional mapping of hand class name → weight (0-1). When
            provided, each hand's weight comes from this dict (defaulting to
            1.0 for hands not in the dict). When None, all hands get 1.0.

        k: Positive combo count or ``ALL``. A class emits no more than its
            post-blocking live availability and divides its mass over the
            number actually emitted.

    Returns: list of (c0, c1, weight, hand_class_name). Fully blocked classes
    retain one weight-zero placeholder. K=1 gives the flat 169-row output
    (one representative combo per class).
    """
    rows, _metadata = _build_hand_grid_range_with_metadata(
        blocked, weights=weights, k=k, board=board,
    )
    return rows


def _expected_hand_grid_rows(
    blocked: set, k: GridK = 1, board: Optional[Sequence[int]] = None,
) -> int:
    """Expected row count, including one placeholder per blocked class."""
    if _grid_k_is_iso(k):
        if board is None:
            raise ValueError("grid K=ISO needs the board to pool suits")
        from app.services.suit_isomorphism import group_by_canonical
        return sum(
            max(1, len(group_by_canonical(
                _hand_class_combos(hand, blocked, k="ALL"), board)))
            for hand in HAND_GRID
        )
    return sum(
        max(1, len(_hand_class_combos(hand, blocked, k=k)))
        for hand in HAND_GRID
    )
