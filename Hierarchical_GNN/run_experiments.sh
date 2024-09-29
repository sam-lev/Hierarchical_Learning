#!/bin/sh

#run_folder_path="/home/sam/Documents/PhD/Research/Hierarchical_Learning/Hierarchical_GNN"

run_base=$(pwd)
run_path="runs"

name="$1"

hst_ablation_experiment() {
	local DATASET=$1
	local DATASUBSET=$2
	local PERSISTENCE=$3
	local LR=$4
	local LREDGE=$5
	local WD=$6
	local RUN_FILE=$7
	local MODEL=$8
	local EXPERIMENT=$9
	# epochs=84 eval_steps=14
	#python main.py --cuda_num=0 --dropout=0.0 --dim_hidden=512 --num_layers=2 \
	#--batch_size=128 --use_batch_norm=False --SLE_threshold=0.9 --N_exp=1 \
	#--dataset="$DATASET" --epochs=84 --homophily=0.9 \
	#--multi_label=True --type_model=HierGNN \
	#--eval_steps=14 --train_by_steps=False \
	#--lr=${LR} --lr2=${LREDGE} --weight_decay=${WD} \
	#--data_subset="$DATASUBSET" --persistence="${PERSISTENCE}" --hier_model="$MODEL" 2>&1 | tee -- "$RUN_FILE"

		# epochs=84 eval_steps=14
	python main.py --cuda_num=0 \
	--dropout_edge=0.0 --dim_hidden_edge=64 \
	--dropout=0.65 --dim_hidden=128 --dim_gin=128 --dim_multiscale_filter_conv=128 --num_layers=3 \
	--batch_size=512 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
	--dataset="$DATASET" --epochs=121 --homophily=0.9 \
	--multi_label=True --type_model=HierGNN \
	--eval_steps=8 --train_by_steps=False \
	--lr=${LR} --lr2=${LREDGE} --weight_decay=${WD} \
	--data_subset="$DATASUBSET" --persistence="${PERSISTENCE}" --hier_model="$MODEL" 2>&1 | tee -- "$RUN_FILE" \
	--experiment="$EXPERIMENT"
}

MODEL="HST"
# best pubmed hst exp_hjt_21_HST_Planetoid_PubMed_LR0.01_LRE_0.01_WD5e-5_PERS0.7,0.9.log
# best for hjt exp_hjt_9_HJT_Planetoid_PubMed_LR3e-4_LRE_0.03_WD5e-8_PERS0.5,0.7,0.9.log
# HJT
# hjty for cora best exp_hjt_8_HJT_Planetoid_Cora_LR3e-3_LRE_0.03_WD1e-12_PERS0.5,0.7,0.9.log
# hst best cora exp_stwo_Planetoid_Cora_LR0.001_LRE_0.01_WD1e-8_PERS0.6.log

for EXPERIMENT in "fixed_init_ablation" "seq_init_ablation"
do
  for DATASET in Planetoid #WikipediaNetwork #WebKB
  do
    for DATASUBSET in  Cora PubMed #roman_empire minesweeper questions amazon_ratings tolokers questions #squirrel  # chameleon # for DATASUBSET in texas # wisconsin cornell
    do
      DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET/EXPERIMENTS"
      mkdir -p "$DATA_RUN_PATH"
      for PERSISTENCE in "0.5,0.7" "0.5,0.7,0.9" "0.5" "0.8"# "0.6,0.9" "0.6,0.8" "0.6,0.9" "0.8,0.9" "0.7" "0.8" "0.9" "0.1" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
      do
        for WD in 5e-5 5e-8 #1e-12
        do
          for LR in 0.01 0.1 0.003 #0.01 #0.03 0.0001
          do
            for LREDGE in 0.01 0.1 0.003 #0.01 #0.03 0.0001 0.01
            do
                  exp_name="exp_${EXPERIMENT}_${MODEL}_${DATASET}_${DATASUBSET}_LR${LR}_LRE_${LREDGE}_WD${WD}_PERS${PERSISTENCE}.log"
                  RUN_FILE="$DATA_RUN_PATH/$exp_name"

                  echo "MODEL=${MODEL}"
                  echo "MODEL=${MODEL}" >> "$RUN_FILE"
                  echo "DATASET=${DATASET}"
                  echo "DATASET=${DATASET}" >> "$RUN_FILE"
                  echo "DATASUBSET=${DATASUBSET}"
                  echo "DATASUBSET=${DATASUBSET}" >> "$RUN_FILE"
                  echo "PERSISTENCE=${PERSISTENCE}"
                  echo "PERSISTENCE=${PERSISTENCE}" >> "$RUN_FILE"
                  echo "LREDGE=${LREDGE}"
                  echo "LREDGE=${LREDGE}" >> "$RUN_FILE"
                  echo "LR=${LR}"
                  echo "LR=${LR}" >> "$RUN_FILE"
                  echo "WD=${WD}"
                  echo "WD=${WD}" >> "$RUN_FILE"
                  echo "EXPERIMENT=${EXPERIMENT}"
                  echo "EXPERIMENT=${EXPERIMENT}" >> "$RUN_FILE"

                  #> "$RUN_FILE"

              hst_ablation_experiment "$DATASET" "$DATASUBSET" "${PERSISTENCE}" "${LR}" "${LREDGE}" "${WD}" "$RUN_FILE" "$MODEL" "$EXPERIMENT"
            done
          done
        done
      done
      DATA_RESULTS_PATH="$run_base/$run_path/$MODEL/RESULTS"
      mkdir -p "$DATA_RESULTS_PATH"
      RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_RESULTS_SUMMARY.log"
      python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
      OPT_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_RESULTS.log"
      python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE"

      OPT_ROC_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_ROC_RESULTS.log"
      python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
    done
  done
done