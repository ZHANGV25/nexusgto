# Card abstraction files

The trainer and the reader both need these two files. They are the published blueprint's own
copies, and the public code rebuilds both byte for byte.

| File | Bytes | What it holds | Rebuilt by |
|---|---|---|---|
| `texture_cache.bin` | 9,322,572 | Flop texture buckets (`TXC1`): 1,755 flop textures, 200 buckets per texture. | The trainer, when no cache is found (slow: over 5 minutes). |
| `turn_centroids.bin` | 2,408 | Turn bucket centroids (`TCN1`): 200 centroids of 3 floats. | The trainer (about 45 s on one core) or `BPSReaderC`, from a fixed seed. |

`SHA256SUMS` lists their hashes. `tests/test_fixture.py` checks them.

The river uses OCHS with 8 opponent clusters. Its two tables (`ochs_clusters.bin` and
`river_centroids.bin`) take under a second to compute, so they are not shipped: the trainer
computes them and caches them in its cache directory (below), and the reader computes them
when it loads a blueprint.

## Where the code looks for them

- The trainer looks in `$BP_CACHE_DIR`, then its working directory, and writes the caches
  it computes to `$BP_CACHE_DIR` (or the working directory when that is unset).
  `solver/precompute/blueprint_worker_unified.py` sets `BP_CACHE_DIR` to its `--output-dir`
  unless it is already set. Put these files there (as `tools/make_fixture.py` does).
- `BPSReaderC` looks next to the `.bps` file, then in `solver/`, then in the working directory.
  Copy both files next to the blueprint you load.
