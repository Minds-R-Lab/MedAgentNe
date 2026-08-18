# Continuous integration

The workflow lives at `docs/ci-workflow.yml` rather than
`.github/workflows/ci.yml`, because the token used for the initial push was
scoped without `workflow` permission. To enable it, copy the file into place:

```bash
mkdir -p .github/workflows
git mv docs/ci-workflow.yml .github/workflows/ci.yml
git commit -m "Enable CI"
git push
```

That push needs a token with the `workflow` scope, or you can add the file
through the GitHub web interface: **Add file → Create new file**, name it
`.github/workflows/ci.yml`, and paste the contents of `docs/ci-workflow.yml`.

## What it does

On every push and pull request, against Python 3.11 and 3.12:

1. Runs the 35 regression tests (`medagentnet/tests`).
2. Runs the harness end to end on the deterministic backend at 80 patients,
   covering experiments E9, E1, E4 and E5.
3. **Fails the build if a headline result regresses**, not merely if something
   errors:
   - full-system F1 below 0.85;
   - removing the cross-departmental assembly step no longer collapsing
     detection to below half the full-system F1;
   - any disallowed field reappearing in a query context;
   - any ground-truth text reappearing in a query context.

The third step is the one that matters. The four defects corrected in this
revision were all silent — the code ran, produced numbers, and the numbers were
wrong. A test suite that only checks for exceptions would have passed on every
one of them.
