# Training metadata

The manifests the 8.05B blueprint's training run wrote at each checkpoint, every 100 million
iterations, as the trainer wrote them.

| File | What it is |
|---|---|
| `meta_100M.json` ... `meta_8B.json` | 80 checkpoint manifests, from 100 million to 8 billion iterations |
| `checkpoint_meta.json` | The run's latest-checkpoint manifest; identical to `meta_8B.json` |
| `SHA256SUMS` | SHA-256 of each file |

The published blueprint stopped at 8.05 billion iterations, after the last checkpoint here; its
own manifest is stored inside the `.bps` file.

## Fields

- `iterations`, `num_info_sets`: info sets grow from 289,192,403 at 100 million iterations to
  993,835,731 at 8 billion.
- `config_hash`: the same in every file, `bc8ac105...`. It hashes the game and abstraction
  settings (see `config_hash` in `solver/precompute/blueprint_worker_unified.py`);
  `configs/fixture.json` trains the test blueprint with the same settings and checks that it
  gets the same hash.
- `hash_table_size`, `hash_load_factor`, `hash_insertion_failures`, `hash_max_probe_observed`:
  the trainer's info-set table (4,000,000,007 slots, a quarter full at the end, no failed
  insertions).
- `time_seconds`, `time_hours`: time since the start of the process that wrote the file, not
  since the start of training. The run was stopped and resumed from its last checkpoint seven
  times, so the series has eight segments; the clock restarts at 700M, 1B, 1.5B, 1.7B, 2.4B,
  4B and 4.3B iterations. `meta_8B.json`'s 69.8 hours is the last segment only. Adding the last
  value of each segment gives 166.1 hours; iterations run after a segment's last checkpoint
  were repeated after the resume and are not counted.
- `dcfr`: empty, so the run used the trainer's default discounting.
