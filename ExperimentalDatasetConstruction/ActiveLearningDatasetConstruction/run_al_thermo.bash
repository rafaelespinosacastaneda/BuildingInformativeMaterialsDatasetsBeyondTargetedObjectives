#!/bin/bash
set -euo pipefail

# -------------------------
# Experiment grid
# -------------------------
RANDOM_SEEDS=(0 )
DATASETS=(ThermoEl)




GROWING_CRITERIA=(
grow_EA_DeltaDiversity
)

# -------------------------
# Target sets
# -------------------------
# 1-outcome targets
TARGETS_1=(electric_cond seebeck_coef zT thermal_cond)

# 2-outcome combos
TARGETS_2=(el_seeb el_zT el_ThC seeb_zT seeb_ThC zT_ThC)

# 3-outcome combos
TARGETS_3=(el_seeb_zT el_seeb_ThC el_zT_ThC seeb_zT_ThC)

# "all outcomes"
TARGETS_ALL=(all)

# -------------------------
# Helper: decide which target list to use for a given growingCriteria
# Prints the list space-separated (so caller can read it into an array)
# -------------------------
targets_for_criterion() {
  local crit="$1"

  case "$crit" in
    # --- single-target methods ---
    grow_QBC|grow_EA_DeltaDiversity)
      echo "${TARGETS_1[*]}"
      ;;

    # --- two-target methods ---
    grow_2QBC|grow_EA_3Obj_TwoTargets|grow_EA_2Obj_TwoTargets_QBC_ONLY)
      echo "${TARGETS_2[*]}"
      ;;

    # --- three-target methods ---
    grow_3QBC|grow_EA_4Obj_ThreeTargets|grow_EA_3Obj_ThreeTargets_QBC_ONLY)
      echo "${TARGETS_3[*]}"
      ;;

    # --- all-target methods ---
    grow_random|grow_multiobj_all)
      echo "${TARGETS_ALL[*]}"
      ;;

    # --- safety ---
    *)
      echo "ERROR: Unknown growingCriteria '$crit'." 1>&2
      exit 1
      ;;
  esac
}

# -------------------------
# Main loop
# -------------------------
for randomSeed in "${RANDOM_SEEDS[@]}"; do
  for growingCriteria in "${GROWING_CRITERIA[@]}"; do
    # Get targets for this criterion
    read -r -a TARGETS <<< "$(targets_for_criterion "$growingCriteria")"

    for dataset in "${DATASETS[@]}"; do
      for target in "${TARGETS[@]}"; do

        outputDir="al/${growingCriteria}/${dataset}/${target}/${randomSeed}"

        if [[ -d "$outputDir" ]]; then
          echo "skipping $outputDir"
          # continue   # uncomment if you want to skip existing runs
        else
          mkdir -p "$outputDir"
        fi

        echo "Running: seed=${randomSeed} crit=${growingCriteria} dataset=${dataset} target=${target}"

        python run_all_thermo.py \
          --growingCriteria "$growingCriteria" \
          --dataset "$dataset" \
          --target "$target" \
          --outputDir "$outputDir" \
          --randomSeed "$randomSeed"

      done
    done
  done
done
