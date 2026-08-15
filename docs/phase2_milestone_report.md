# Phase 2 Baseline Milestone — Report

**Run:** 2026-08-14. **Target:** 2022-23 season, Gameweek 20 (a real historical gameweek, chosen because that season is grade-A audited in `docs/vaastav_archive_audit.md` and predates defensive contributions, so this baseline's known DC gap can't distort the result). **Reproduce:** `PYTHONPATH=src python scripts/run_phase2_milestone.py`.

## What this proves

Per spec Part LXVI, this is the required first milestone: a complete, mechanically-connected pipeline running on real (not mock) data, with a recommendation frozen to disk *before* actual results are revealed, then scored against those actual results.

```
real historical fixtures (177, strictly before GW20)
    -> attack/defense team model (ported from World Cup repo)
    -> Dixon-Coles scoreline distribution (12 target-GW fixtures)
    -> naive minutes model (699 players, 6-GW lookback)
    -> proportional goal/assist allocation
    -> Monte Carlo simulation (converged at 15,000 sims/player, tol=0.05)
    -> deterministic FPL scoring engine (2026/27 rules config)
    -> legal squad optimizer (scipy MILP: budget/quota/club-limit)
    -> starting XI + captain optimizer (formation-legal MILP)
    -> FROZEN artifact (artifacts/phase2_milestone/gw20_recommendation.json, hashed, source-data-hashed)
    -> reveal actual GW20 results
    -> score frozen recommendation against reality
    -> compare against a recent-form baseline
```

## Result

| | Projected | Realized (actual) |
|---|---|---|
| Model-driven squad (11 starters + captain bonus) | 66.17 | **81** |
| Recent-form baseline (same optimizer, EP = sum of last-6-GW actual points) | — | 50 |

The model squad beat the recent-form baseline by +31 realized points. The captain pick (Demarai Gray, projected highest at 7.30 EP) actually returned only 5 — the win came from strong depth (James Ward-Prowse 15, Morgan Gibbs-White 10, Kieran Trippier 9, Martin Ødegaard 9), not from the top projection landing.

## What this result does NOT prove

**n = 1 gameweek is not statistical evidence of anything.** One gameweek beating one baseline could easily be variance — football is exactly the high-variance domain the spec repeatedly warns about (Part XXXIX: protect against research overfitting; don't treat a single favorable outcome as validation). This milestone's purpose was to prove the pipeline *runs end-to-end on real data and produces a falsifiable, frozen, scoreable output* — not to demonstrate forecasting skill. That requires Phase 3's full historical replay across many gameweeks and multiple seasons, with proper scoring rules (log loss, calibration, CRPS) and statistical significance testing (paired bootstrap, block-by-gameweek), not a single point comparison.

## Known limitations baked into this baseline (all documented in code, repeated here for visibility)

- **No defensive contributions, no BPS/bonus simulation.** Bonus points are excluded from *projected* points entirely (they're not in the simulation); the *realized* comparison uses the archive's true `total_points` (which does include real bonus), so the projected-vs-realized gap partly reflects this asymmetry, not just forecast error. This is a known, documented limitation — not hidden in the gap.
- **Team model constants are unfit.** `K_BASE=0.045`, `HALFLIFE_DAYS=380` (attack/defense decay) and `rho=-0.04` (Dixon-Coles) are carried over from the World Cup repo's international-football fit, explicitly *not* refit for club football. Refitting via ablation is Phase 4 work.
- **Minutes model is the simplest possible baseline** (recent start-rate only) — no injury news, no rotation/fixture-congestion awareness, no predicted-lineup signal.
- **Proportional goal/assist allocation** is exactly the approach spec Part X names as something to move beyond, kept only because it's the correct Phase 2 starting point.
- **No cross-player correlation beyond shared match scorelines** — no joint BPS ranking, no competing-penalty-taker modelling, no auto-substitution logic applied to the realized score.
- **Squad optimizer is single-gameweek only** — no transfers, no chips, no multi-gameweek horizon (Phase 7/8/9 work).

## Reproducibility

The frozen recommendation records: `generated_at`, full model config (team/scoreline/minutes/attacking model identifiers and hyperparameters), simulation seed and simulation count actually run, and SHA-256 hashes of all three source data files used. The evaluation artifact is a separate file (`gw20_evaluation.json`), written only after the recommendation file already existed — the recommendation itself was never edited after actual results were revealed, per spec Part LXIII's prohibition on retroactive editing.

## Test coverage added alongside this milestone

31 tests total (up from 14 before this session's Phase 2 work), including:
- A property test proving the scoring engine reconstructs **100% of a real season's `total_points`** (26,505/26,505 rows, 2022-23) — found and fixed a real YAML structural bug in the process.
- A property test proving the Monte Carlo simulator's vectorized fast-path scoring never drifts from the authoritative scoring engine — found and fixed a real bug (clean-sheet minutes gating) in the process.
- Independent legality verification for the optimizer (budget, quotas, club limits, formation bounds, captain-must-be-a-starter), including an infeasibility test.

## Next step

Phase 3: generalize this single-gameweek run into deadline-by-deadline sequential replay across a full season (or multiple), so the "does this actually work" question can be answered with statistical evidence rather than one data point.
