#!/bin/bash

MARKER=$1

source $GITHUB_WORKSPACE/test_env/bin/activate
pytest -m $MARKER -vv -ra --durations=0 --durations-min=0.001 | tee report_${MARKER}.txt; test ${PIPESTATUS[0]} -eq 0 || exit 1
echo "# Timing profile of ${MARKER}" >> $GITHUB_STEP_SUMMARY
python $GITHUB_WORKSPACE/.github/build_job_summary.py report_${MARKER}.txt >> $GITHUB_STEP_SUMMARY
rm report_${MARKER}.txt