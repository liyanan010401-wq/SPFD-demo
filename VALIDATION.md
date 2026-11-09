# Validation

Validated on 2026-11-09 using the existing Windows Python 3.12 CPU environment. This was an execution check, not a fresh-environment dependency installation test.

```sh
python -B -X utf8 -m unittest discover -s tests -v
python -B -X utf8 run_demo.py --output outputs/publication_check
```

- All 13 tests passed.
- The complete demo run exited successfully and produced synthetic diagnostics and predictions.
- The execution reported `publication_evidence: false`.
- Laboratory implementation, laboratory weights and real sonar samples are not included.

Dependency versions are recorded in `requirements-tested.txt`. Synthetic execution success does not validate the paper's experimental results.
