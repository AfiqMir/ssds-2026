# Repository Experiment Rules

## API_TEST holdout quarantine

Treat `API_TEST/` as a final external holdout, never as experiment input.

- Do not read files under `API_TEST/` during feature engineering, training,
  hyperparameter tuning, model selection, blend selection, or threshold tuning.
- Select and freeze candidates using only the four rolling-origin OOF folds and
  the documented leave-one-fold-out/stress-fold criteria.
- Before running `API_TEST`, write the pre-API decision and selected parameters
  to the experiment's `experiment_summary.json`.
- Run `API_TEST` only after the candidate is frozen. Treat its score as a
  one-way transfer check, not permission to change the candidate.
- Do not retune, reweight, or select a follow-up variant in response to an API
  score. Any follow-up hypothesis must be justified from train/OOF diagnostics
  alone and must not use the prior API result as its objective.
- Avoid repeated API evaluation of near-identical variants. Prefer one frozen
  primary candidate and, only when OOF evidence independently supports it, one
  predeclared aggressive candidate.
- Report API results separately from OOF selection metrics and state explicitly
  that they were observed after freezing.

These rules also apply to notebooks and ad-hoc analysis commands, not only to
checked-in experiment scripts.
