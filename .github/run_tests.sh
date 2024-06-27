#!/bin/bash

MARKER=$1
NOVIRTUALENV=$2

# Check if the second argument is provided and if it is equal to --no-virtual-env
if [ -z "$NOVIRTUALENV" ] || [ "$NOVIRTUALENV" != "--no-virtual-env" ]; then
  source $GITHUB_WORKSPACE/test_env/bin/activate
fi

pytest -m "$MARKER" -vv -ra --durations=0 --durations-min=0.001 | tee report.txt; test ${PIPESTATUS[0]} -eq 0 || exit 1
echo "# Timing profile of ${MARKER}" >> $GITHUB_STEP_SUMMARY
python $GITHUB_WORKSPACE/.github/build_job_summary.py report.txt >> $GITHUB_STEP_SUMMARY
rm report.txt
