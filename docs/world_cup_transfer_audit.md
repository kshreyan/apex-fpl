# World Cup Predictor → APEX FPL Transfer Audit

**Audited:** 2026-08-14
**Source repo:** `~/Documents/fifa-world-cup-predictor` (not under version control; 15 Python modules, 3,506 LOC, `data/raw|processed|outputs`)
**Method:** every file listed below was read in full (not inferred from README or column names). Classifications use the scheme mandated by the spec: A = directly reusable, B = reusable after modification, C = conceptually useful only, D = present but unvalidated, E = missing, F = unsafe/incorrect.

## Scope of the source project

`fifa-world-cup-predictor` forecasts **international-tournament match outcomes** (scoreline distributions, W/D/L, bracket advancement) for the FIFA World Cup 2026, using ~49k historical international results (`martj42/international_results`, GitHub, CC-licensed). It has **no player-level modelling of any kind** — no minutes, positions, individual goals/assists, cards, saves, or any fantasy-scoring concept. It is purely a team-strength → scoreline → tournament-bracket pipeline. This bounds the entire audit: only Layer A (team match environment) of APEX FPL's problem decomposition has anything to inherit from it.

## Component-by-component findings

### `src/elo.py` — chronological Elo ratings
**Exists.** Base 1500, new-team prior 1300, home advantage +65 (non-neutral matches only), tournament-importance-weighted K-factor (WC=60, WC-qualifiers=40, continental=45, friendly=20, other=30), FIFA-style margin-of-victory multiplier (1.0 / 1.5 / (11+gd)/8 for gd 0-1 / 2 / 3+).
**Validated:** `compute_elo` streams matches in strict date order and only ever reads `ratings` state accumulated from strictly earlier rows — no forward references. Confirmed by direct code read, not by trusting the "leak-free" docstring claim.
**Classification: A** for the *mechanism* (chronological online rating update), **B** for the *parameters* (1500/1300/±65/tournament weights are calibrated for international football with sparse per-team match counts; Premier League clubs play ~38+ league matches/season with much denser data, so K-factor and decay constants need PL-specific refitting, not reuse verbatim).
**Reuse recommendation:** port the update mechanism (`expected()`, `margin_multiplier()`, streaming loop structure) into `src/apex_fpl/models/teams/elo.py`; re-fit all constants against Premier League historical results.

### `src/time_decay_elo.py` — separate attack/defense ratings with recency decay
**Exists.** Online gradient-style update in log-rate space, separate attack (`a`) and defense (`d`) per team, 730-day half-life decay toward neutral when a team hasn't played recently, tournament-importance weighting reused from `elo.py`.
**Validated:** Same streaming-order guarantee verified by reading `compute_attack_defense`; decay is applied *before* computing each match's expected goals, not after, so no lookahead.
**Classification: A** (mechanism), **B** (730-day half-life is tuned for teams that play a handful of times per year across multiple competitions — a PL club plays weekly, so a much shorter half-life, or an explicit competition-segmented rating, will likely be needed; this must be decided by ablation, not assumption, per the spec's anti-dogma principle).
**Reuse recommendation:** this is the strongest candidate for APEX FPL's core team-strength signal (expected goals for/against per fixture), since attack/defense separation is exactly what player-goal-allocation and clean-sheet models need downstream.

### `src/models.py`, `src/models_v2.py` — scoreline and result models
**Exists.** Poisson regression for expected goals; Dixon-Coles low-score correction (rho fit by bounded MLE); bivariate Poisson via Karlis-Ntzoufras shared-component decomposition (with negative-binomial overdispersion and zero-inflation diagnostics reported, not just assumed); classifier zoo (logistic regression, random forest, XGBoost, LightGBM, CatBoost) blended via a **time-ordered holdout stacking meta-learner** (the 85/15 split inside `train_stack` is chronological, not random — verified in code).
**Validated:** `data/outputs/backtest_report.md` and `data/outputs/backtesting_report.txt` show real, unfavorable-looking (i.e., not cherry-picked) numbers: log loss 0.90–1.07, result accuracy 50–61%, exact-scoreline accuracy 9–14% across 2014/2018/2022 WC folds, ensemble beating a logistic-Elo baseline only marginally. This honesty (explicitly stated in the README as "the ensemble's edge over a strong Elo baseline is modest") is itself evidence the backtest wasn't tuned against these folds after the fact.
**Classification: A** for the modelling *techniques* (Dixon-Coles, bivariate Poisson, stacking pattern, scoreline-matrix construction `score_matrix`/`bp_matrix`), **E** for anything player-level — team goal *totals* are the ceiling of what this code produces; translating a team's expected goals into which *player* scores requires an entirely new model (Part X of the spec) that does not exist here in any form.
**Reuse recommendation:** port `score_matrix`, `bp_matrix`, `dc_tau`, and the stacking harness structure into `src/apex_fpl/models/teams/`; refit every learned parameter (Poisson coefficients, rho, classifier hyperparameters) on Premier League fixtures — the World Cup artifacts (`model_artifacts*.joblib`) are useless as-is since PL and international-tournament goal-scoring environments differ (league table incentives, squad rotation, differing opposition strength distributions).

### `src/calibration_utils.py` — probability calibration and scoring
**Exists.** Log loss, multiclass Brier, per-class Brier, RPS (ranked probability score for ordered H/D/A), expected calibration error (ECE), draw-specific calibration check, isotonic and Platt (sigmoid) per-class calibrators with renormalization, reliability-diagram plotting.
**Validated:** used live in `models_v2.backtest()` and `main()` — calibration method (isotonic vs Platt vs none) is chosen by comparing log loss on a held-out *tail* slice of training data, not on the evaluation folds themselves. This is methodologically correct (calibration fit must not touch the test set) and directly matches spec Part XX's requirement.
**Classification: A.** This is the single most directly reusable module: proper scoring rules and calibration are position-agnostic and format-agnostic. FPL will need the same functions applied far more broadly — binary calibration for start-probability/60-minute-probability/clean-sheet-probability, and multiclass calibration is a strict special case already implemented.
**Reuse recommendation:** generalize `ProbabilityCalibrator` to accept an arbitrary number of classes (currently hardcoded to 3 via `IDX = {"H":0,"D":1,"A":2}`) and port into `src/apex_fpl/calibration/`. Non-trivial modification, so **B** rather than pure **A** once the 3-class hardcoding is accounted for — noting this precisely rather than rounding up to "fully reusable."

### `src/market.py` — betting-market consensus
**Exists.** Correct odds math: `p_raw = 1/decimal_odds`, proportional de-vig (`p_raw / sum(p_raw)`), cross-bookmaker consensus computed on the **logit scale** (mean of `log(p/(1-p))` per outcome, then renormalized) rather than a naive probability average — this is the more defensible method the spec itself calls out in Part XI's data-source guidance.
**Validated — partially.** The math is exercised in `models_v2.backtest()` when odds data exists, but **no live odds were ever supplied**: `data/raw/bookmaker_odds_manual_template.csv` contains only `is_example=True` placeholder rows, and `load_market()` correctly returns `None` in that case rather than fabricating a market signal. The backtest report explicitly logs "`market_only` / `ml_plus_market` are pending" — confirmed honest non-fabrication, verified by reading both `market.py`'s filtering logic and the actual CSV contents.
**Classification: A** (de-vig + logit-consensus math), **E** (a working odds feed — none exists; this is a data-sourcing problem, not a code problem).
**Reuse recommendation:** port `implied()`, `devig_proportional()`, `logit()`/`inv_logit()`, `consensus_from_books()` verbatim into `src/apex_fpl/models/teams/market.py`. Building a real, ToS-compliant odds adapter for Premier League markets is separate Phase 1/3 work (Part IV data-source registry).

### `src/data_cleaning.py`, `src/feature_engineering.py` — cleaning and leak-free feature construction
**Exists.** Name standardization, duplicate removal, chronological sort enforcement, `result`/`total_goals`/`goal_diff` derivation, rolling-window features (5/10-match form, goals for/against, win rate, clean-sheet rate) all built with `.shift(1)` before `.rolling()` — i.e., the match itself is excluded from its own rolling features, verified directly in `rolling_features()`.
**Validated.** `data/outputs/leakage_audit_report.md` is a real generated artifact (not hand-written prose) that programmatically checks for target-token substrings in the feature list and reports zero hits; it also states plainly that market/squad/injury data is "attached ONLY to 2026 rows; never back-filled onto historical training rows" — checked against the code in `team_strength.py::load_squad_template()` and confirmed correct.
**Classification: B.** The shift-then-roll pattern and the general discipline (one chronologically-sorted timeline, explicit "known by" ordering, an automated leakage-audit generator) transfer directly as a *pattern* APEX FPL must replicate for gameweek-level features, but none of the code operates on FPL's actual entities (players, positions, fixtures-with-2-legs-per-gameweek, price data) so it needs to be rewritten against new schemas, not just copy-pasted.
**Reuse recommendation:** treat this as the reference implementation for `tests/leakage/` design and for the Bronze/Silver/Gold snapshot discipline (Part V), not as an importable module.

### `src/simulate_tournament.py`, `src/simulate_v2.py` — Monte Carlo tournament simulation
**Exists.** Fully vectorized (NumPy, no per-simulation Python loops) 50k/100k-run Monte Carlo of the *48-team FIFA bracket format specifically* — group tables with correct FIFA tiebreakers, backtracking constraint search for third-place-team bracket slot assignment (verified against all 495 group combinations per the README, and the backtracking logic in `assign_thirds()` reads correctly), Elo-weighted penalty-shootout tie resolution.
**Validated.** Runs against real (frozen) fixture data and produces `wc2026_tournament_simulation.csv` with sane, monotonically-decreasing round-advancement probabilities.
**Classification: C — conceptually useful only.** The *vectorization pattern* (simulate all N runs of a stochastic process as NumPy array operations rather than a Python loop-of-loops) is the one thing worth carrying forward; everything else — bracket structure, group tiebreakers, shootout resolution — is FIFA-tournament-specific and has no analog in a Premier League gameweek, where the actual simulation need is a **jointly correlated per-gameweek player-event Monte Carlo** (Part XVI of the spec), a fundamentally different object (players within a match, not teams within a bracket).
**Reuse recommendation:** do not port this code. Reimplement the vectorization *approach* for `src/apex_fpl/simulation/`.

### `src/wc2026_config.py`
**Exists.** Static 2026 World Cup structural facts (groups, bracket wiring, name-standardization map, FIFA ranking snapshot). **Classification: N/A — not transferable in any form**, it's tournament-specific trivia. No action.

### `src/fetch_current_odds.py`, `src/team_strength.py` (squad-value wiring), manual CSV templates
**Exists** as data-adapter patterns (the-odds-api.com fetcher; manual-template-with-integrity-flags pattern for data that can't be pulled programmatically, using `is_example`/`is_real_data`/`data_status` columns so placeholder rows can never silently enter a "real" aggregate).
**Classification: C.** The *pattern* — never let a template placeholder be mistaken for real data, enforce it with an explicit flag column checked by the loader — is worth replicating for APEX FPL's own manual-input needs (injuries, predicted lineups). The specific fetchers are World-Cup/odds-API-specific and not reusable as code.

## What is verifiably absent (Part II required this explicitly)

Checked for and **not found anywhere in the repository** (grepped all 15 source files and all output artifacts):
- Any minutes/appearance model
- Any player-level event model (goals/assists/shots/xG at player granularity)
- Any FPL-specific assist, bonus-points, or defensive-contribution logic
- Any goalkeeper save/BPS logic
- Any price, ownership, or competitive-field model
- Any squad-selection or transfer optimizer (no OR-Tools/CP-SAT/HiGHS/MIP code of any kind)
- Any chip-valuation logic
- Any prediction-versioning, immutable-forecast-freezing, or artifact-hashing infrastructure (all outputs are overwritten in place on each run — there is no `run_id`, no hash, no timestamp column in any output CSV)
- Git version control of any kind

## Net conclusion

Reusable: the team-strength statistical core (Elo → attack/defense ratings → Dixon-Coles/bivariate-Poisson scoreline generation → calibration → time-based backtest/leakage-audit discipline). This becomes the seed for `src/apex_fpl/models/teams/` and `src/apex_fpl/calibration/`, after refitting every learned constant on Premier League data. Everything from Layer A's minutes model onward — which is most of the FPL problem — is new build with no existing code to draw on.
