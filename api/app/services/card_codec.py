"""Pure card/label codec helpers: a SUPPORT module with no state.

Card<->int and hand-class encoding, and solver-label<->action-index mapping.
They are pure functions, so ``app.engine.*`` can depend on this leaf module
without pulling in anything stateful.

Convention (matches the C library):
  - card int 0-51: ``rank * 4 + suit`` where rank is 0=2 .. 12=A
    (i.e. ``12 - RANKS.index(char)``) and suit index is c<d<h<s.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

# ── Constants ───────────────────────────────────────────────────────────

RANKS = "AKQJT98765432"
SUIT_ORDER = ("c", "d", "h", "s")


# ── Card <-> int encoding ───────────────────────────────────────────────


def _rank_index(r: str) -> int:
    """Rank character → 0-12 (A=0, K=1, ..., 2=12)."""
    return RANKS.index(r.upper())


def _card_to_int(card: str) -> int:
    """Card string → int 0-51, matching solver convention."""
    rank = 12 - _rank_index(card[0])
    suit = SUIT_ORDER.index(card[1].lower())
    return rank * 4 + suit


def _int_to_card(c: int) -> str:
    """Int 0-51 → card string."""
    rank_char = RANKS[12 - (c >> 2)]
    suit_char = SUIT_ORDER[c & 3]
    return rank_char + suit_char


def _hand_class(c0: int, c1: int) -> str:
    """Map a specific combo (c0, c1) to its 169-class name (e.g. 'AKs')."""
    r0 = 12 - (c0 >> 2)  # rank index in RANKS (0=A, 12=2)
    s0 = c0 & 3
    r1 = 12 - (c1 >> 2)
    s1 = c1 & 3

    rank_hi = RANKS[min(r0, r1)]
    rank_lo = RANKS[max(r0, r1)]

    if r0 == r1:
        return f"{rank_hi}{rank_lo}"  # pair, e.g. "AA"
    if s0 == s1:
        return f"{rank_hi}{rank_lo}s"
    return f"{rank_hi}{rank_lo}o"


# ── Solver label <-> action index mapping ───────────────────────────────


def _action_to_solver_label(
    action: str,
    amount: Optional[float],
    pot: float,
    bet_sizes: Tuple[float, ...],
) -> str:
    """Map an API action to a solver label for strategy lookup.

    `bet_sizes` is the blueprint's training-time postflop_bet_sizes vocab,
    read from .bps metadata at provider init. The solver label space
    derives from this vocab: e.g. for a (0.75,) vocab, a "raise" maps to
    "Bet 75%"; for (0.5, 1.0), it would snap to the nearer of
    "Bet 50%" or "Bet 100%".
    """
    # Lazy import to avoid a card_codec -> hand_grid import edge at module
    # load.
    from app.services.hand_grid import _snap_amount_to_tree

    if action == "check":
        return "Check"
    if action == "call":
        return "Call"
    if action == "allin":
        return "All-in"
    if action == "raise":
        if amount is None or amount <= 0:
            # Pick the smallest tree size as the "raise" default when no
            # amount is supplied. This follows the Pluribus open-size
            # convention and works with any vocab without a magic
            # 75% literal.
            smallest = min(bet_sizes)
            return f"Bet {int(round(smallest * 100))}%"
        return _snap_amount_to_tree(amount, pot, bet_sizes)
    return action


def _solver_label_to_action_index(
    label: str,
    has_to_call: bool,
    bet_sizes: Tuple[float, ...],
    add_explicit_allin: bool,
) -> int:
    """Map a solver label to the integer action index used by
    compute_action_hash in mccfr_blueprint.c.

    The order matches generate_actions() in mccfr_blueprint.c for the
    current blueprint's vocab:

      has_to_call:  [fold=0, call=1, raise_size_1=2, ..., raise_size_N=1+N,
                     (allin=2+N if add_explicit_allin)]
      not:          [check=0, bet_size_1=1, ..., bet_size_N=N,
                     (allin=1+N if add_explicit_allin)]

    `bet_sizes` is the trainer's postflop_bet_sizes (sorted ascending here,
    defensively, in case metadata isn't pre-sorted).
    `add_explicit_allin` mirrors the trainer flag; when it is 0 there is no
    all-in slot at this level.

    For an unknown label, returns a sensible fallback: check (no
    has_to_call) or call (has_to_call).
    """
    sizes = sorted(bet_sizes)
    label_lower = label.lower().strip()
    is_allin = "all-in" in label_lower or "allin" in label_lower

    if not has_to_call:
        if label_lower == "check":
            return 0
        if is_allin:
            if not add_explicit_allin:
                # Trainer didn't emit an allin action here; snap to the
                # nearest large bet (top of vocab) so the action_hash
                # at least stays on-tree.
                return len(sizes)
            return 1 + len(sizes)
        # Bet sized — find matching size from the vocab.
        idx = _match_size_label_to_idx(label_lower, sizes)
        if idx is None:
            return 0  # fallback to check
        return 1 + idx
    else:
        if label_lower == "fold":
            return 0
        if label_lower == "call":
            return 1
        if is_allin:
            if not add_explicit_allin:
                return 1 + len(sizes)
            return 2 + len(sizes)
        idx = _match_size_label_to_idx(label_lower, sizes)
        if idx is None:
            return 1  # fallback to call
        return 2 + idx


def _match_size_label_to_idx(
    label_lower: str, sizes: List[float]
) -> Optional[int]:
    """Find the index of the bet/raise size in `sizes` whose percentage
    appears in `label_lower`.

    Labels look like "bet 75%", "raise 100%". The integer percentage is
    matched to the closest size in the vocab. Returns None if no
    percentage can be parsed.
    """
    m = re.search(r"(\d+)%", label_lower)
    if m is None:
        return None
    pct = int(m.group(1))
    best_idx = 0
    best_err = float("inf")
    for i, s in enumerate(sizes):
        err = abs(int(round(s * 100)) - pct)
        if err < best_err:
            best_err = err
            best_idx = i
    return best_idx
