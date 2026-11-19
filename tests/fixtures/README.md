# Test fixtures

`fixture.bps` (283,339 bytes) is a tiny blueprint of the published game: the same preflop tiers,
postflop sizes, river OCHS and trainer settings as the 8.05B blueprint, so its `config_hash`
equals the 8.05B's. It was trained for 1,000 iterations on one thread with a 2,000,003-slot
hash table and holds 25,243 information sets. Its strategies are far from converged; use it to
check that code loads, looks up and plays a real blueprint, never for strategy values.

Rebuild it with `python tools/make_fixture.py` (Linux, after building the trainer). The settings
are in `configs/fixture.json`, and `tests/test_fixture.py` checks the bytes.
