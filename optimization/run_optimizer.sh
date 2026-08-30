#!/usr/bin/env bash
set -euo pipefail

export GUROBI_HOME=/opt/gurobi1201/linux64
export GRB_LICENSE_FILE=/opt/gurobi1201/gurobi.lic

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
output_dir="$repo_root/data/results/ensemble_weight_optimizer"
arguments=("$@")
for ((index = 0; index < ${#arguments[@]}; index++)); do
	if [[ "${arguments[index]}" == "--output-dir" ]]; then
		((index + 1 < ${#arguments[@]})) || { echo "Missing value for --output-dir." >&2; exit 2; }
		output_dir="${arguments[index + 1]}"
		break
	fi
done

julia --project="$repo_root/optimization" "$repo_root/optimization/run.jl" "$@"
"$HOME/.virtualenvs/classifier/bin/python" "$repo_root/scripts/util/write_ensemble_optimizer_report.py" \
	--output-dir "$output_dir" --remove-source-csvs
