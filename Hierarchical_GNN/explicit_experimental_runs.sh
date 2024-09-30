#!/bin/sh

#run_folder_path="/home/sam/Documents/PhD/Research/Hierarchical_Learning/Hierarchical_GNN"

run_base=$(pwd)
run_path="runs"

name="$1"

hjt_ablation_experiment() {


	DATA_RUN_PATH="$run_base/$run_path/HJT/chameleon/EXPERIMENTS_HJT"
  mkdir -p "$DATA_RUN_PATH"
	exp_name="exp_${EXPERIMENT}_${MODEL}_${DATASET}_${DATASUBSET}_LR${LR}_LRE_${LREDGE}_WD${WD}_PERS${PERSISTENCE}.log"
  RUN_FILE="$DATA_RUN_PATH/$exp_name"

	PERSISTENCE="0.6,0.8"
	python main.py --cuda_num=0 \
	--dropout_edge=0.0 --dim_hidden_edge=128 \
	--dropout=0.65 --dim_hidden=128 --dim_gin=128 --dim_multiscale_filter_conv=128 --num_layers=3 \
	--batch_size=512 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
	--dataset=WikipediaNetwork --epochs=121 --homophily=0.9 \
	--multi_label=True --type_model=HierGNN \
	--eval_steps=8 --train_by_steps=False \
	--lr=0.001 --lr2=0.001 --weight_decay=1e-6 \
	--data_subset=chameleon --persistence="${PERSISTENCE}" --hier_model=HJT 2>&1 | tee -- "$RUN_FILE" \
	--experiment=None

	DATA_RUN_PATH="$run_base/$run_path/HJT/Cora/EXPERIMENTS_HJT"
  mkdir -p "$DATA_RUN_PATH"
	exp_name="exp_${EXPERIMENT}_${MODEL}_${DATASET}_${DATASUBSET}_LR${LR}_LRE_${LREDGE}_WD${WD}_PERS${PERSISTENCE}.log"
  RUN_FILE="$DATA_RUN_PATH/$exp_name"
	PERSISTENCE="0.6,0.8"
	python main.py --cuda_num=0 \
	--dropout_edge=0.0 --dim_hidden_edge=128 \
	--dropout=0.65 --dim_hidden=128 --dim_gin=128 --dim_multiscale_filter_conv=128 --num_layers=3 \
	--batch_size=512 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
	--dataset=Planetoid --epochs=121 --homophily=0.9 \
	--multi_label=True --type_model=HierGNN \
	--eval_steps=8 --train_by_steps=False \
	--lr=0.001 --lr2=0.001 --weight_decay=1e-6 \
	--data_subset=Cora --persistence="${PERSISTENCE}" --hier_model=HJT 2>&1 | tee -- "$RUN_FILE" \
	--experiment=None
}

hjt_ablation_experiment
