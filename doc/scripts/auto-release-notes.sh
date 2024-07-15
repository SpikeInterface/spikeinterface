#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: $0 START_DATE END_DATE [LABEL] [BRANCH1,BRANCH2] [LIMIT]"
    exit 1
fi

START_DATE="$1"
END_DATE="$2"

if [ -z "$3" ] || [ "$3" = "all" ]; then
    LABELS=("core" "extractors" "preprocessing" "sorters" "postprocessing" "qualitymetrics" "curation" "widgets" "generators" "hybrid" "sortingcomponents" "motion correction" "documentation" "continuous integration" "packaging" "testing")
else
    LABELS=("$3")
fi

if [ -n "$4" ]; then
    IFS=',' read -ra BRANCHES <<< "$4"
else
    BRANCHES=("main")
fi

if [ -n "$5" ]; then
    LIMIT=$5
else
    LIMIT=300
fi

for LABEL in "${LABELS[@]}"; do
    echo "$LABEL:"
    echo ""
    for BRANCH in "${BRANCHES[@]}"; do
        gh pr list --repo SpikeInterface/spikeinterface --limit $LIMIT --label "$LABEL" --base "$BRANCH" --state merged --json number,title,mergedAt \
            | jq -r --arg start_date "${START_DATE}T00:00:00Z" --arg end_date "${END_DATE}T00:00:00Z" \
            '.[] | select(.mergedAt >= $start_date and .mergedAt <= $end_date) | "* \(.title) (#\(.number))"'
    done
    echo ""
done

echo "Contributors:"
echo ""
gh pr list --repo SpikeInterface/spikeinterface --limit 1000 --base main --state merged --json number,title,author,mergedAt \
  | jq -r --arg start_date "${START_DATE}T00:00:00Z" --arg end_date "${END_DATE}T00:00:00Z" \
  '[.[] | select(.mergedAt >= $start_date and .mergedAt <= $end_date) | .author.login] | unique | .[] | "* @" + .'
