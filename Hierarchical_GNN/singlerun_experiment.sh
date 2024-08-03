#!/bin/sh

#run_folder_path="/home/sam/Documents/PhD/Research/Hierarchical_Learning/Hierarchical_GNN"

run_base=$(pwd)
run_path="runs"

name="$1"

experiment() {
	local DATASET=$1
	local DATASUBSET=$2
	local PERSISTENCE=$3
	local LR=$4
	local LREDGE=$5
	local WD=$6
	local RUN_FILE=$7
	local MODEL=$8
	# epochs=84 eval_steps=14
	python main.py --cuda_num=0 --dropout=0.2 --dim_hidden=512 --num_layers=5 \
	--batch_size=64 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
	--dataset="$DATASET" --epochs=21 --homophily=0.9 \
	--multi_label=True --type_model=HierGNN \
	--eval_steps=7 --train_by_steps=False \
	--lr=${LR} --lr2=${LREDGE} --weight_decay=${WD} \
	--data_subset="$DATASUBSET" --persistence="${PERSISTENCE}" --hier_model="$MODEL" 2>&1 | tee -- "$RUN_FILE"
}

###########################################
#        
#                          Hierarchical SUCCESIVE Training Experiments
# 
###########################################

###########################################
MODEL="HJT"


for DATASET in WebKB
do
	for DATASUBSET in texas #wisconsin cornell texas #Wisconsin squirrel squirrel_filtered_directed roman_empire
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.6,0.9" # "0.5" "0.6" "0.7" "0.8" "0.9" "0.6,0.8" "0.8,0.9" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for LREDGE in 0.01 # 0.03 0.003 0.001 #0.01 #0.03 0.0001
			do
				for LR in 0.03 # 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001
				do
					for WD in 1e-8 # 0 1e-4 1e-6 1e-8 #1e-12
					do
			    			exp_name="exp_${name}_${MODEL}_${DATASET}_${DATASUBSET}_LR${LR}_LRE_${LREDGE}_WD${WD}_PERS${PERSISTENCE}.log"
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
			    			
			    			#> "$RUN_FILE"
			    			
						experiment "$DATASET" "$DATASUBSET" "${PERSISTENCE}" "${LR}" "${LREDGE}" "${WD}" "$RUN_FILE" "$MODEL"
					done
				done
			done
		done
		DATA_RESULTS_PATH="$run_base/$run_path/$MODEL/RESULTS"
		mkdir -p "$DATA_RESULTS_PATH"
		RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_RESULTS_SUMMARY.log"
		python print_log_file_results.py "$DATA_RESULTS_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE" 
		
		OPT_ROC_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done
