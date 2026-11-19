# The BPS3 blueprint format

A `.bps` file holds a trained blueprint: the average strategy for every information set the
trainer visited, plus a JSON manifest that describes the game it was trained in. The trainer
writes it (`_stream_bps_export` in `solver/precompute/blueprint_worker_unified.py`, which streams
the bytes from `bp_export_strategies` in `solver/src/blueprint/mccfr_blueprint.c`). Two readers
load it: the C reader (`solver/src/blueprint/bps_reader.c`, driven by `BPSReaderC` in
`solver/python/engine/bps_reader_py.py`) and the preflop extractor
(`solver/precompute/extract_preflop_json.py`).

`tests/fixtures/fixture.bps` is a small example. Every integer is little-endian.

## File layout

| Offset | Size | Type | Field |
|---|---|---|---|
| 0 | 4 | bytes | Magic `BPS3` |
| 4 | 8 | u64 | `compressed_size`: length of the strategy payload in bytes |
| 12 | 4 | u32 | `meta_size`: length of the manifest in bytes |
| 16 | `compressed_size` | bytes | Strategy payload, one LZMA stream (`.xz` container, preset 1) |
| 16 + `compressed_size` | `meta_size` | bytes | Manifest, UTF-8 JSON |
| 16 + `compressed_size` + `meta_size` | rest | bytes | Optional `BPR3` action-EV section (below) |

The manifest sits after the payload, so a reader that needs only the manifest seeks past the
payload and never decompresses it (`BPSReaderC._load_metadata_only`).

## Strategy payload

Decompressed, the payload is a 12-byte header followed by one record per (information set,
bucket) pair.

| Size | Type | Field |
|---|---|---|
| 4 | bytes | Magic `BPS3` |
| 4 | u32 | `num_entries`: number of records |
| 4 | u32 | `num_players` (6) |

Each record:

| Size | Type | Field |
|---|---|---|
| 1 | u8 | `player`: seat of the player to act, 0 to 5 |
| 1 | u8 | `street`: 0 preflop, 1 flop, 2 turn, 3 river |
| 2 | u16 | `bucket`: the hand's bucket on this street (below) |
| 8 | u64 | `board_hash`: always 0 (the bucket already accounts for the board) |
| 8 | u64 | `action_hash`: hash of the betting so far (below) |
| 1 | u8 | `na`: number of legal actions, at most 16 |
| `na` | u8 × `na` | Quantized average strategy, one byte per action |

A record is 21 + `na` bytes. Records come in the trainer's hash-table order, not sorted.

The strategy bytes are `round(p × 255)` of the trainer's average strategy (`strategy_sum`
normalized; regret matching when no sum was accumulated), clamped to 0 to 255. Readers
renormalize each row to sum to 1, and a row of zeros reads as uniform. A bucket with no record
at a node that exists reads as `255 / na` per action, which is also uniform.

Actions are listed in the order the trainer generates them (`generate_actions` in
`mccfr_blueprint.c`): fold, check or call, then the bet or raise sizes from the manifest, smallest
first. The game model reproduces the same order (`legal_action_categories` in
`api/app/engine/agent/game_state.py`), and `api/app/engine/agent/preflop_line_compiler.py` checks
the Python preflop against the C generator.

## Keys

A node is identified by `(action_hash, player, street)`, and each node holds one row per bucket.

- `action_hash` hashes the action history from the first preflop action to the current decision,
  as the index of each chosen action in its node's action list (0 for the first action, and so on).
  Chance events (cards) are not part of the history. Starting from `h = 0xFEDCBA9876543210`, each
  action index `a` updates `h = combine(h, a × 17 + 3)`. With the manifest's
  `"hash_mixer": "splitmix64"`, `combine(h, x) = mix64(h ^ mix64(x))`, where `mix64` is the
  splitmix64 finalizer (`x ^= x >> 30; x *= 0xbf58476d1ce4e5b9; x ^= x >> 27;
  x *= 0x94d049bb133111eb; x ^= x >> 31`). Blueprints without `hash_mixer`, or with `"boost"`,
  use the older combiner `h ^= x + 0x9e3779b97f4a7c15 + (h << 6) + (h >> 2)`. Both are in
  `compute_action_hash` in `solver/src/blueprint/bps_reader.c`.
  `api/app/engine/agent/key_builder.py` builds the history from a game state.
- `bucket` on the preflop street is one of 169 hand classes, numbered from the highest pair down:
  AA is 0, then AKs 1, AKo 2, AQs 3 and so on to A2o, then KK, KQs, KQo, ..., down to the pair of twos at 168
  (the `class_map` loop in `bp_init_ex`). Cards are encoded `rank × 4 + suit`, with ranks 2 to A
  as 0 to 12 and suits c, d, h, s as 0 to 3.
- `bucket` after the flop is one of `postflop_buckets` (200) buckets. The flop bucket comes from
  the texture cache, the turn bucket from the turn centroids (both in `data/abstraction/`), and
  the river bucket from OCHS with 8 opponent clusters. `bps_hand_to_bucket` in
  `solver/src/blueprint/bps_reader_bucketize.c` computes them the same way the trainer does, and
  `solver/tests/test_flop_bucket_alignment.c` checks that the two agree.

## Manifest

The manifest is a JSON object. These fields are written by the trainer and read by the code:

| Field | Meaning | Read by |
|---|---|---|
| `type` | `unified_blueprint` for a final export, `phase_r_strategy_snapshot` for a snapshot taken during training | |
| `num_players`, `blinds`, `initial_stack` | The game: 6 players, blinds 50 and 100 chips, 10,000-chip stacks | preflop extractor |
| `preflop_tiers`, `preflop_max_raises` | Preflop raise sizes as pot fractions per raise level (`"0"`, `"1"`, `"2"`), and the raise cap | preflop extractor, game model |
| `preflop_bet_sizes` | The first tier's sizes, kept for older readers | |
| `postflop_bet_sizes`, `subsequent_bet_sizes` | Postflop first-bet sizes and the sizes of later raises, as pot fractions | search, rollouts |
| `add_explicit_allin` | 1 if all-in is a separate action; 0 means a player is all-in only when a menu size reaches their stack | preflop extractor, search |
| `preflop_buckets`, `postflop_buckets` | Bucket counts per street (169 and 200) | C reader |
| `hash_mixer` | `splitmix64` or `boost`; which combiner built `action_hash` | C reader |
| `iterations`, `num_info_sets` | Training iterations and stored information sets | |
| `config_hash` | SHA-256 of the game and abstraction settings (`config_hash` in `blueprint_worker_unified.py`); equal hashes mean the same game and abstraction | tests |
| `hash_table_size`, `hash_load_factor`, `hash_insertion_failures`, `hash_max_probe_observed` | The trainer's information-set table at export | |
| `dcfr` | Discounting overrides; empty means the trainer's default schedule | |
| `schema_version`, `has_action_evs` | 3 and `true` only when a `BPR3` section follows | C reader |

Snapshots also carry depth-limit fields (`depth_limit_cont_strats` and related) and the
snapshot's source; readers ignore fields they don't use. The training checkpoints in
`data/training/` use the same field names.

## Optional action-EV section

When the manifest has `schema_version` 3 or more and `has_action_evs` set, a `BPR3` section
follows the manifest: the magic `BPR3`, a u64 compressed length, then an LZMA stream. Decompressed,
it is the magic `BPR3`, a u32 record count, and records of `player` (u8), `street` (u8),
`bucket` (u16), `action_hash` (u64), `na` (u8) and `na` float32 action values in chips. The
trainer in this repository doesn't write this section; the C reader still accepts it.

## Reader cache

`BPSReaderC` writes `<file>.bps.cache` next to the blueprint after its first full load, and later
loads read the cache instead of decompressing the payload. The cache is the reader's in-memory
table on disk; it is not part of the format, and deleting it only makes the next load slower.
