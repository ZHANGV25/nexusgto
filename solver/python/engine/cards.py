"""Card encoding helpers shared by the search code and its tests.

Card ints are rank * 4 + suit, with ranks 2..A = 0..12 and suits c, d, h, s = 0..3.
"""

RANK_MAP = {'2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
            '9': 7, 'T': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12}
SUIT_MAP = {'c': 0, 'd': 1, 'h': 2, 's': 3}
RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"


def card_to_int(card_str: str) -> int:
    """Convert card string like 'Ah' to integer 0-51."""
    rank = RANK_MAP.get(card_str[0], -1)
    suit = SUIT_MAP.get(card_str[1], -1)
    if rank < 0 or suit < 0:
        raise ValueError(f"Invalid card: {card_str}")
    return rank * 4 + suit


def int_to_card(card_int: int) -> str:
    """Convert integer 0-51 to card string like 'Ah'."""
    return RANK_CHARS[card_int >> 2] + SUIT_CHARS[card_int & 3]


def parse_range_string(range_str: str) -> list:
    """Parse PioSOLVER-format range string to list of (card0, card1, weight) tuples.

    Supports:
      - Specific combos: "AhKh" -> [(Ah, Kh, 1.0)]
      - Weighted: "AhKh:0.5" -> [(Ah, Kh, 0.5)]
      - Hand groups: "AKs" -> all 4 suited AK combos
      - Comma-separated: "AA,KK,AKs"
    """
    hands = []
    if not range_str:
        return hands

    for part in range_str.split(","):
        part = part.strip()
        if not part:
            continue

        weight = 1.0
        if ":" in part:
            part, w = part.rsplit(":", 1)
            try:
                weight = float(w)
            except ValueError:
                pass

        if len(part) == 4:
            # Specific combo: AhKs
            c0 = card_to_int(part[:2])
            c1 = card_to_int(part[2:])
            if c0 > c1:
                c0, c1 = c1, c0
            hands.append((c0, c1, weight))
        elif len(part) == 3:
            # Hand group: AKs, AKo, AA
            r0 = RANK_MAP.get(part[0], -1)
            r1 = RANK_MAP.get(part[1], -1)
            if r0 < 0 or r1 < 0:
                continue
            if part[2] == 's':
                for suit in range(4):
                    c0 = r0 * 4 + suit
                    c1 = r1 * 4 + suit
                    if c0 != c1:
                        hands.append((min(c0, c1), max(c0, c1), weight))
            elif part[2] == 'o':
                for s0 in range(4):
                    for s1 in range(4):
                        if s0 != s1:
                            c0 = r0 * 4 + s0
                            c1 = r1 * 4 + s1
                            hands.append((min(c0, c1), max(c0, c1), weight))
        elif len(part) == 2:
            # Pair: AA
            r0 = RANK_MAP.get(part[0], -1)
            r1 = RANK_MAP.get(part[1], -1)
            if r0 == r1 and r0 >= 0:
                for s0 in range(4):
                    for s1 in range(s0 + 1, 4):
                        c0 = r0 * 4 + s0
                        c1 = r1 * 4 + s1
                        hands.append((c0, c1, weight))

    # Deduplicate
    seen = set()
    unique = []
    for c0, c1, w in hands:
        key = (min(c0, c1), max(c0, c1))
        if key not in seen:
            seen.add(key)
            unique.append((key[0], key[1], w))
    return unique
