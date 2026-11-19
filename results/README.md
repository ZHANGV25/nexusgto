# Results

Every published number, the per-deal data behind it, and a script that
recomputes it.

| File | What it is |
|---|---|
| `numbers.yaml` | One row per published number: value, SE, 95% interval, deals, the data files and their SHA-256, and `reproducible: true` or `false` (with the reason) |
| `reproduce.py` | Recomputes every reproducible row from the data and checks it at the precision it was published. Standard library only |
| `all_arms.csv` | Every search configuration measured against the blueprint field, including the ones whose data is lost |
| `MANIFEST.sha256` | SHA-256 of each data file |
| `data/` | The data files. Not in git: they are the release asset below |

## Reproduce

Download `nexusgto-results.tar` from the v1.0.0 release into the repository root, then:

```
mkdir -p results/data
tar -xf nexusgto-results.tar -C results/data
python results/reproduce.py
```

The first lines are the headline:

```
Headline: Every-street search with own-range tracking vs the blueprint field
  14,250 deals, +5.73 ± 1.53 bb/100 (95% CI 2.74 to 8.72)  [matches numbers.yaml]
```

It takes about 10 seconds. `--only ID ...` checks single rows. Exit code 1
means a number differs.

## Release asset

`nexusgto-results.tar`: 71 files, 49.3 MB (the files inside are already gzipped).
SHA-256 of the tar:
`defc6f237298f57c5e5edd4969645e169153c89e6cdef21d4239a67d2f13e449`

| Files | Contents |
|---|---|
| `match.<arm>.<field>.deal_groups.jsonl.gz` | One row per deal: the deal, the six seat rotations, the tested player's net chips per hand (raw and AIVAT-adjusted) |
| `match.<arm>.<field>.summaries.jsonl.gz` | One row per job (750 deals): pooled estimates, search timings, fallback counts, the settings used |
| `match.<arm>.<field>.mix_compare.jsonl.gz` | Per decision: the search's action mix next to the blueprint's |
| `match.<arm>.<field>.fallback_spots.jsonl.gz` | Every search that fell back to the blueprint, with the reason |
| `lbr.<arm>.hands.jsonl.gz` | Local best response: one row per deal and seat |
| `lbr.<arm>.summaries.jsonl.gz` | Per seed: the LBR settings and estimates |
| `decision_audit.*.gz` | The 1,200 audited turn and river spots, their scores per configuration, and the solve logs |

## Conventions

- Stacks 100bb, no rake; 100 chips = 1 big blind, so chips per hand = bb/100.
- A deal is played six times, once per seat rotation, against five copies of
  the field. The deal is the independent unit; the SE uses the n-1 variance;
  the 95% interval is the mean ± 1.96 SE.
- Paired differences pair the same (seed, deal) in two runs and check that the
  cards match.
- AIVAT-adjusted is the default; `metric: raw` rows are without it.
- LBR averages the two seat arms of a deal, so the deal is the unit there too.

## Arms

| Arm | Search settings (as recorded in each summary) |
|---|---|
| `blueprint` | No search: the 8.05B blueprint by table lookup |
| `blueprint_rerun` | The same, seeds 9101-9104 of the nit and maniac fields, rerun on 18 Nov 2026 with this release's code (see What is lost) |
| `first_config` | Postflop re-solve with 2 or 3 players under a 3 s budget, ranges from preflop actions only |
| `flop_only` | `NEXUS_FLOP_CHANCE_SOLVE=1`, `NEXUS_VEC_SAMPLED_OPP_HAND=root`, `--production-live-streets flop` |
| `flop_only_combo_prior` | `flop_only` plus `NEXUS_RANGE_COMBO_PRIOR=1` |
| `all_streets` | `NEXUS_FLOP_CHANCE_SOLVE=1`, `NEXUS_VEC_SAMPLED_OPP_HAND=root`, `NEXUS_RANGE_COMBO_PRIOR=1`, `NEXUS_RANGE_POSTFLOP_CONDITIONING=1` |
| `all_streets_own_range` | `all_streets` plus `--hero-own-reach` (LBR: `--hero-range own`) |
| `all_streets_serving_flags_40k`, `all_streets_serving_candidate` | Later candidates: `all_streets` plus latency controls from a branch that is not in this release (see `all_arms.csv`) |

Every search arm also ran with `NEXUS_LEAF_VEC_SAMPLED=1`,
`NEXUS_HU_LEAF_MIRROR=1`, `NEXUS_OFFTREE_ADD=1` and
`NEXUS_MAX_PROJECTED_FILL_BYTES=7000000`, and 8 search threads.

`production` in flags and data names the search configuration under test.
Arm ids are part of the data file names, so they stay as recorded.

## Flag map

| Flag | Effect | Read in |
|---|---|---|
| `NEXUS_LEAF_VEC_SAMPLED=1` | Depth-limited leaves are valued per visit by sampled rollouts inside the vector traverser; no leaf table is built | `api/app/engine/cpu_es_solver.py` |
| `NEXUS_VEC_SAMPLED_OPP_HAND=root` | At such a leaf, opponent hands are drawn from their root ranges instead of the traversal's sampled hand | `api/app/engine/cpu_es_solver.py` |
| `NEXUS_FLOP_CHANCE_SOLVE=1` | A heads-up flop root is solved through the turn and river in-tree, over 8 sampled turn cards and 6 river cards per turn card | `api/app/engine/cpu_es_solver.py` |
| `NEXUS_RANGE_COMBO_PRIOR=1` | Range rows carry each hand class's weight per physical combo (the deal prior) | `api/app/engine/agent/live_resolve_adapter.py` |
| `NEXUS_RANGE_POSTFLOP_CONDITIONING=1` | Each player's range is conditioned on the blueprint's probability of the postflop actions that player took | `api/app/engine/agent/postflop_conditioning.py` |
| `NEXUS_RIVER_AVG_READOUT=1` | River solves are read out as the average strategy (earlier arms only) | `api/app/engine/cpu_es_solver.py` |
| `NEXUS_HU_LEAF_MIRROR=1` | Heads-up leaves query the blueprint's positionally matching continuation | `api/app/engine/agent/live_resolve_adapter.py` |
| `NEXUS_OFFTREE_ADD=1` | An opponent's off-tree bet size is added to the re-solve tree | `api/app/engine/agent/offtree_add.py` |
| `NEXUS_MAX_PROJECTED_FILL_BYTES` | A solve whose projected table size exceeds this falls back to the blueprint | `api/app/engine/cpu_es_solver.py` |
| `--hero-own-reach` | Own-range tracking: the tested player's own range comes from the strategies its searches played | `api/scripts/run_match_cli.py`, `api/scripts/own_reach.py` |
| `--production-live-streets` | Search only on the listed streets; play the blueprint elsewhere | `api/scripts/run_match_cli.py` |

## What changed in the repack

- Run names became the public arm and field ids above.
- Floats were rounded to 6 decimals. Every published estimate is unchanged
  (the headline is 5.730752 ± 1.526676 before and after).
- Machine paths were removed from summaries and LBR configs.
- Run logs, queue logs, server environment files, a 3-deal pilot and the
  decision audit's solved tables (`.npz`) were left out.
- `first_config` pools two runs of the same configuration: seeds 9101-9104
  and 9105-9108.

## What is lost

Most of the results before the every-street runs were stored only in cloud
storage that was deleted on 5 Oct 2026. Their numbers survive as text; their
per-deal data does not. `numbers.yaml` marks each such row
`reproducible: false`, and `all_arms.csv` marks each such arm
`data_available: no`. One arm, `all_streets_serving_flags_40k`, survives for
11,250 of its 14,250 deals.

The blueprint's own runs in the nit and maniac fields lost seeds 9101-9104.
They were rerun on 18 Nov 2026 with this release's code and are shipped as
the `blueprint_rerun` arm. The blueprint by lookup has no search randomness,
so a rerun should give the same hands; to check that, seeds 9105-9108, whose
files survive, were rerun too and matched them deal by deal (6,000 deals,
323,206 decisions, no difference). The rerun's pooled figures for 9101-9104
(nit +32.26 ± 2.11, maniac +58.53 ± 10.24) equal the figures recorded for the
lost runs, and the two "vs the blueprint playing the same seat" rows now
reproduce.

Four published figures are slightly off, and `numbers.yaml` keeps both the
published and the corrected value (`corrected_*`, with a note):

- the LBR gap's SE is 21.8, not 21.3 (21.3 treated the two seats of a deal as
  independent);
- one 95% upper bound is 9.58, not 9.59;
- one fallback cause has 311 spots, not 310;
- the 4 Oct blueprint LBR rerun's SE is 21.0, not 21.1.
