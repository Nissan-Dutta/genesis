#!/bin/sh
# Re-download the two ProteinGym benchmark tables this analysis uses, pinned to one commit
# (2026-03-25), and check they are byte-identical to the copies the results were computed on.
set -e
cd "$(dirname "$0")"
COMMIT=144fe22b07dfaeec2b366f2346203a9838a55b4c
B=https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/$COMMIT/benchmarks/DMS_zero_shot/substitutions/Spearman
curl -fsSLO "$B/DMS_substitutions_Spearman_DMS_level.csv"
curl -fsSLO "$B/Summary_performance_DMS_substitutions_Spearman.csv"
shasum -a 256 -c SHA256SUMS
