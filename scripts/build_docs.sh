#!/usr/bin/env bash
# Regenerate the HTML API documentation (docs/) and the test report (reports/).
# Run from the repository root:  bash scripts/build_docs.sh
set -euo pipefail

# API documentation from the docstrings (Google style), one page per module.
python -m pdoc src/rasher_inspection --docformat google --output-directory docs

# Test report with line coverage. Tests run on a small fake dataset,
# so no download, GPU, PyTorch or Ultralytics is needed.
mkdir -p reports
python -m pytest --html=reports/test_report.html --self-contained-html \
       --cov=rasher_inspection --cov-report=term \
       --junitxml=reports/junit.xml | tee reports/test_summary.txt
