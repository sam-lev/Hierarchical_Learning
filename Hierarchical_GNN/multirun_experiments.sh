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
	--data_subset="$DATASUBSET" --persistence="${PERSISTENCE}" --hier_model="$MODEL" 2>&1 | tee -- "$RUN_FILE"
}

###########################################
#        
#                          Hierarchical SUCCESIVE Training Experiments
# 
###########################################

###########################################
MODEL="HJT"
: << COMMENT
for DATASET in WikipediaNetwork
do
	for DATASUBSET in chameleon squirrel  
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.1" "0.9" "0.5" "0.6" "0.7" "0.8" "0.6,0.8" "0.6,0.9" "0.8,0.9" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for LREDGE in 0.003 0.001 0.01 0.03  # 0.01 #0.03 0.0001
			do
				for LR in 0.003 0.03 0.003 0.01 0.001  #0.001 #0.01 #0.03 0.0001 #0.01 0.03 0.003
				do
					for WD in 5e-3 1e-6 1e-8 0#1e-12
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
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE" 
		
		OPT_ROC_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done



for DATASET in WebKB
do
	for DATASUBSET in cornell wisconsin texas #Wisconsin squirrel squirrel_filtered_directed roman_empire
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.9" "0.1" "0.9" "0.5" "0.6" "0.7" "0.9" "0.6,0.8" "0.6,0.9" "0.8,0.9" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for LREDGE in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001
			do
				for LR in 0.0003 0.03 0.003 0.01 0.001 #0.01 #0.03 0.0001
				do
					for WD in 1e-4 0 1e-6 0 1e-4 #1e-12
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
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE" 
		
		OPT_ROC_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done
COMMENT

########################################################################################





: << COMMENT
#
#           planetoid
#
for DATASET in Planetoid
do
	for DATASUBSET in CiteSeer #Cora PubMed
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.5,0.7,0.9" "0.3,0.9" "0.5,0.9" "0.6,0.8" "0.6,0.9" "0.8,0.9" "0.7" "0.8" "0.9" "0.1" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for LR in  3e-4 3e-3 0.03 1e-4 1e-3 0.01 #0.01 #0.03 0.0001
			do
				for LREDGE in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001 0.01
				do
					for WD in 5e-5 5e-8 5e-4 0 1e-12 #1e-12
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
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE"

		OPT_ROC_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done
COMMENT

MODEL="HST"

for DATASET in HeterophilousGraphDataset #Planetoid #WikipediaNetwork #WebKB
do
	for DATASUBSET in  roman_empire minesweeper questions amazon_ratings tolokers questions #squirrel  # chameleon # for DATASUBSET in texas # wisconsin cornell    
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.6,0.95" "0.6,0.8,0.95" "0.5,0.9" "0.6,0.9" "0.6,0.8" "0.6,0.9" "0.8,0.9" "0.7" "0.8" "0.9" "0.1" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for WD in 3e-8 3e-6 5e-4 0 1e-12 #1e-12
			do
				for LR in 3e-3 3e-4 0.03 1e-4 0.01 1e-3 #0.01 #0.03 0.0001 
				do
					for LREDGE in 0.003 0.03 0.3 0.01 0.03 0.001 #0.01 #0.03 0.0001 0.01
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
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE" 
		
		OPT_ROC_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done

: << COMMENT
#
#        WikipediaNetwork
#
for DATASET in WikipediaNetwork
do
	for DATASUBSET in chameleon squirrel    
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.7,0.8,0.9" "0.8,0.9" "0.6,0.9" "0.6,0.8" "0.6,0.9" "0.8,0.9" "0.7" "0.8" "0.9" "0.1" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for LR in  3e-4 3e-3 0.03 1e-4 1e-3 0.01 #0.01 #0.03 0.0001
			do
				for LREDGE in 0.003 0.01 0.03 0.001 #0.01 #0.03 0.0001 0.01 
				do
					for WD in 5e-6 5e-8 5e-4 0 1e-12 #1e-12
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
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE" 
		
		OPT_ROC_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done


#
#           planetoid
#
for DATASET in Planetoid 
do
	for DATASUBSET in CiteSeer #Cora PubMed  
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.5,0.7,0.9" "0.3,0.9" "0.5,0.9" "0.6,0.8" "0.6,0.9" "0.8,0.9" "0.7" "0.8" "0.9" "0.1" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for LR in  3e-4 3e-3 0.03 1e-4 1e-3 0.01 #0.01 #0.03 0.0001
			do
				for LREDGE in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001 0.01 
				do
					for WD in 5e-5 5e-8 5e-4 0 1e-12 #1e-12
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
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE" 
		
		OPT_ROC_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done

##########################
#        switching to HST
##########################
MODEL="HST"

for DATASET in WikipediaNetwork
do
	for DATASUBSET in squirrel #chameleon 
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.5" "0.6" "0.7" "0.8" "0.9" "0.6,0.8" "0.6,0.9" "0.8,0.9" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for LREDGE in 0.01 0.03 0.003 0.001 # 0.01 #0.03 0.0001
			do
				for LR in 0.01 0.03 0.003 0.001  #0.001 #0.01 #0.03 0.0001 #0.01 0.03 0.003
				do
					for WD in 0 1e-4 1e-6 1e-8 #1e-12
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
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE" 
		
		OPT_ROC_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done
COMMENT
###########################################
#        
#                          Hierarchical SUCCESIVE Training Experiments
# 
###########################################
MODEL="HJT"


for DATASET in WebKB
do
	for DATASUBSET in wisconsin cornell texas #Wisconsin squirrel squirrel_filtered_directed roman_empire
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.5" "0.6" "0.7" "0.8" "0.9" "0.6,0.8" "0.6,0.9" "0.8,0.9" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for LREDGE in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001
			do
				for LR in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001
				do
					for WD in 0 1e-4 1e-6 1e-8 #1e-12
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
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE" 
		
		OPT_ROC_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done

for DATASET in HeterophilousGraphDataset 
do
	for DATASUBSET in roman_empire minesweeper questions amazon_ratings tolokers #Wisconsin squirrel squirrel_filtered_directed roman_empire
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.5" "0.6" "0.7" "0.8" "0.9" "0.6,0.8" "0.6,0.9" "0.8,0.9" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for LREDGE in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001
			do
				for LR in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001
				do
					for WD in 0 1e-4 1e-6 1e-8 #1e-12
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
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE" 
		
		OPT_ROC_RESULTS_FILE="$DATA_RESULTS_PATH/$DATASUBSET_${name}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done



###########################################
#        
#                          Hierarchical SUCCESIVE Training Experiments
# 
###########################################
MODEL="HJT"


for DATASET in WebKB
do
	for DATASUBSET in wisconsin cornell texas #Wisconsin squirrel squirrel_filtered_directed roman_empire
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.5" "0.6" "0.7" "0.8" "0.9" "0.6,0.8" "0.6,0.9" "0.8,0.9" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for LREDGE in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001
			do
				for LR in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001
				do
					for WD in 0 1e-4 1e-6 1e-8 #1e-12
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
		RESULTS_FILE="$DATA_RUN_PATH}/${DATASUBSET}_RESULTS_SUMMARY.log"
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RUN_PATH}/${DATASUBSET}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE" 
		
		OPT_ROC_RESULTS_FILE="$DATA_RUN_PATH}/${DATASUBSET}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done

: << COMMENT
for DATASET in Planetoid 
do
	for DATASUBSET in Cora 
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.7" "0.8" "0.9" "0.6,0.8" "0.6,0.9" "0.8,0.9" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for LREDGE in 0.01 0.03 0.003 0.001 #0.0001 #0.01 #0.03 0.0001
			do
				for LR in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001
				do
					for WD in 0 1e-4 1e-6 1e-8 #1e-12
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
		RESULTS_FILE="$DATA_RUN_PATH}/${DATASUBSET}_RESULTS_SUMMARY.log"
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RUN_PATH}/${DATASUBSET}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE" 
		
		OPT_ROC_RESULTS_FILE="$DATA_RUN_PATH}/${DATASUBSET}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done

#for DATASET in Planetoid WebKB WebKB WebKB Planetoid Planetoid HeterophilousGraphDataset HeterophilousGraphDataset HeterophilousGraphDataset HeterophilousGraphDataset #chameleon_directed chameleon_filtered_directed squirrel squirrel_filtered_directed roman_empire minesweeper questions amazon_ratings tolokers
#do
#	for DATASUBSET in Cora cornell texas wisconsin CiteSeer PubMed questions minesweeper amazon_ratings tolokers #Wisconsin squirrel squirrel_filtered_directed roman_empire
#	do
#		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
#		mkdir -p "$DATA_RUN_PATH"
#		for PERSISTENCE in "0.5" "0.6" "0.7" "0.8" "0.9" "0.6,0.8" "0.6,0.9" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
#		do
#			for LREDGE in 0.01 0.03 0.003 0.001 0.0001 #0.01 #0.03 0.0001
#			do
#				for LR in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001
#				do
#					for WD in 0 1e-4 1e-6 1e-8 #1e-12
#					do
#			    			exp_name="exp_${name}_${MODEL}_${DATASET}_${DATASUBSET}_LR${LR}_LRE_${LREDGE}_WD${WD}_PERS${PERSISTENCE}.log"
#			    			RUN_FILE="$DATA_RUN_PATH/$exp_name"
#			    			echo "MODEL=${MODEL}"
#			    			echo "MODEL=${MODEL}" >> "$RUN_FILE"			    			
#			    			echo "DATASET=${DATASET}"
#			    			echo "DATASET=${DATASET}" >> "$RUN_FILE"
#			    			echo "DATASUBSET=${DATASUBSET}"
#			    			echo "DATASUBSET=${DATASUBSET}" >> "$RUN_FILE"
#			    			echo "PERSISTENCE=${PERSISTENCE}"
#			    			echo "PERSISTENCE=${PERSISTENCE}" >> "$RUN_FILE"
#			    			echo "LREDGE=${LREDGE}"
#			    			echo "LREDGE=${LREDGE}" >> "$RUN_FILE"
#			    			echo "LR=${LR}"
#			    			echo "LR=${LR}" >> "$RUN_FILE"
#			    			echo "WD=${WD}"
#			    			echo "WD=${WD}" >> "$RUN_FILE"
#			    			
#			    			#> "$RUN_FILE"
#			    			
#						experiment "$DATASET" "$DATASUBSET" "${PERSISTENCE}" "${LR}" "${LREDGE}" "${WD}" "$RUN_FILE" "$MODEL"
#					done
#				done
#			done
#		done
#		RESULTS_FILE="$DATA_RUN_PATH}/${DATASUBSET}_RESULTS_SUMMARY.log"
#		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
#	done
#done
#list_of_floats_values=("0.1,0.2,0.3" "1.1,1.2,1.3" "2.1,2.2,2.3")
#for i in "${!float_values[@]}"; do
#    run_python_script "${float_values[$i]}" 

for DATASET in Planetoid 
do
	for DATASUBSET in PubMed CiteSeer Cora
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.5" "0.6" "0.7" "0.8" "0.9" "0.6,0.8" "0.6,0.9" "0.8,0.9" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for LREDGE in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001 0.01 
			do
				for LR in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001
				do
					for WD in 0 1e-4 1e-6 1e-8 #1e-12
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
		RESULTS_FILE="$DATA_RUN_PATH}/${DATASUBSET}_RESULTS_SUMMARY.log"
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RUN_PATH}/${DATASUBSET}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE" 
		
		OPT_ROC_RESULTS_FILE="$DATA_RUN_PATH}/${DATASUBSET}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done

for DATASET in Planetoid 
do
	for DATASUBSET in CiteSeer PubMed
	do
		DATA_RUN_PATH="$run_base/$run_path/$MODEL/$DATASUBSET"
		mkdir -p "$DATA_RUN_PATH"
		for PERSISTENCE in "0.5" "0.6" "0.7" "0.8" "0.9" "0.6,0.8" "0.6,0.9" "0.8,0.9" #"0.1,0.2,0.3" #"0.9,0.8" "0.9,0.7" "0.8,0.7" "0.9,0.3" "0.8,0.3"
		do
			for LREDGE in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001
			do
				for LR in 0.01 0.03 0.003 0.001 #0.01 #0.03 0.0001
				do
					for WD in 0 1e-4 1e-6 1e-8 #1e-12
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
		RESULTS_FILE="$DATA_RUN_PATH}/${DATASUBSET}_RESULTS_SUMMARY.log"
		python print_log_file_results.py "$DATA_RUN_PATH" 2>&1 | tee -- "$RESULTS_FILE"
		OPT_RESULTS_FILE="$DATA_RUN_PATH}/${DATASUBSET}_OPT_RESULTS.log"
		python best_test_accuracy.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_RESULTS_FILE" 
		
		OPT_ROC_RESULTS_FILE="$DATA_RUN_PATH}/${DATASUBSET}_OPT_ROC_RESULTS.log"
		python best_roc_auc.py "$RESULTS_FILE" 2>&1 | tee -- "$OPT_ROC_RESULTS_FILE"
	done
done
COMMENT
