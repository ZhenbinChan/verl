#!/bin/bash
# ============================================================
# Aggregate lighteval QA results from multiple tasks into one table.
#
# Usage:
#   bash aggregate_qa_results.sh [OUTPUT_DIR]
#
# Scans all results_*.json files under OUTPUT_DIR/results/ recursively,
# merges them, and prints a summary table (markdown + plain text).
# ============================================================

OUTPUT_DIR="${1:-/home/chenzhb/Workspaces/verl/eval_output/Qwen2.5-1.5B}"

# ----------------------------------------------------------
# Find all results files recursively under the results/ tree
# ----------------------------------------------------------
shopt -s nullglob globstar
RESULTS_DIR="$OUTPUT_DIR/results"
FILES=("$RESULTS_DIR"/**/results_*.json)
shopt -u nullglob globstar

# Also try without the "results" subfolder (in case lighteval outputs directly)
if [ ${#FILES[@]} -eq 0 ]; then
    shopt -s nullglob
    FILES=("$OUTPUT_DIR"/results_*.json)
    shopt -u nullglob
fi

if [ ${#FILES[@]} -eq 0 ]; then
    echo "ERROR: No results_*.json files found under $OUTPUT_DIR"
    echo "Run evaluations first, then re-run this script."
    exit 1
fi

echo "==> Found ${#FILES[@]} results file(s) under $OUTPUT_DIR"

# ----------------------------------------------------------
# Step 1: Merge all result files into one JSON array
#   Each element: { task: "namespace|taskname", metrics: {...} }
# ----------------------------------------------------------
jq -s '
  def normalize_task:
    split("|") | .[0] + "|" + .[1];

  (map(.results // {}) | add)
  | to_entries
  | map({ task: (.key | normalize_task), metrics: .value })
  | sort_by(.task)
' "${FILES[@]}" > /tmp/lighteval_qa_merged.json

# Extract metric names as a JSON array
METRICS_ARR=$(jq -c '[.[].metrics | keys[]] | unique' /tmp/lighteval_qa_merged.json)
FIRST_METRIC=$(jq -r '.[0] // empty' <<< "$METRICS_ARR")

if [ -z "$FIRST_METRIC" ]; then
    echo "ERROR: No metrics found in results files."
    rm -f /tmp/lighteval_qa_merged.json
    exit 1
fi

METRICS_LINE=$(jq -r '.[]' <<< "$METRICS_ARR" | tr '\n' ' ')
echo "==> Metrics: $METRICS_LINE"

# ----------------------------------------------------------
# Step 2: Markdown table
# ----------------------------------------------------------
echo ""
echo "### QA Evaluation Results"
echo ""

# Header
printf "| %-45s" "Task"
jq -r '.[] | " | \(.)"' <<< "$METRICS_ARR" | tr -d '\n'
echo " |"

# Separator
printf "|%s" " --- "
jq -r '.[] | " | --- "' <<< "$METRICS_ARR" | tr -d '\n'
echo " |"

# Data rows
jq -r --argjson cols "$METRICS_ARR" '
  def fmt_val:
    if . == null then "-"
    elif type == "number" then (. * 10000 | round / 10000 | tostring)
    elif type == "object" or type == "array" then (. | tostring)
    else tostring
    end;

  .[] | . as $row |
  "| \(.task)" +
  ($cols | map(" | \($row.metrics[.] | fmt_val)") | join("")) +
  " |"
' /tmp/lighteval_qa_merged.json

# ----------------------------------------------------------
# Step 3: Plain-text table (TSV -> column)
# ----------------------------------------------------------
echo ""
echo "================================================================================"
echo ""

{
    printf "Task"
    jq -r '.[] | "\t\(.)"' <<< "$METRICS_ARR" | tr -d '\n'
    echo ""

    jq -r --argjson cols "$METRICS_ARR" '
      def fmt_val:
        if . == null then "-"
        elif type == "number" then (. * 10000 | round / 10000 | tostring)
        elif type == "object" or type == "array" then (. | tostring)
        else tostring
        end;

      .[] | . as $row |
      "\(.task)" +
      ($cols | map("\t\($row.metrics[.] | fmt_val)") | join(""))
    ' /tmp/lighteval_qa_merged.json
} | column -t -s $'\t'

rm -f /tmp/lighteval_qa_merged.json
