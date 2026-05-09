"""
Validation harness for the BurnTheLies NVI gate pipeline.

This package holds:
  - ground_truth_labels.json: labeled cluster outcomes used as fixtures
    (primarily regression snapshots with a small set of human-verified labels)
  - schema.json: JSON Schema validating the fixture format
  - evaluate.py: harness that runs `compute_nvi()` against the live DB and
    compares each cluster's actual gate trace and NVI cap to the expected
    values.

Run from the repo root with:
    python -m intelligence.validation
"""
