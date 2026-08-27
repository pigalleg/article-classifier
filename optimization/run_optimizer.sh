#!/usr/bin/env bash
set -euo pipefail

export GUROBI_HOME=/opt/gurobi1201/linux64
export GRB_LICENSE_FILE=/opt/gurobi1201/gurobi.lic

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec julia --project="$repo_root/optimization" "$repo_root/optimization/run.jl" "$@"
