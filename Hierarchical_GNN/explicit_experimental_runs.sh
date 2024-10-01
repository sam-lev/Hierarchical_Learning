#!/bin/sh

#run_folder_path="/home/sam/Documents/PhD/Research/Hierarchical_Learning/Hierarchical_GNN"

run_base=$(pwd)
run_path="runs"

name="_FiltVal_exp_"

# best pubmed hst exp_hjt_21_HST_Planetoid_PubMed_LR0.01_LRE_0.01_WD5e-5_PERS0.7,0.9.log
# best for hjt exp_hjt_9_HJT_Planetoid_PubMed_LR3e-4_LRE_0.03_WD5e-8_PERS0.5,0.7,0.9.log
# HJT
# hjty for cora best exp_hjt_8_HJT_Planetoid_Cora_LR3e-3_LRE_0.03_WD1e-12_PERS0.5,0.7,0.9.log
# hst best cora exp_stwo_Planetoid_Cora_LR0.001_LRE_0.01_WD1e-8_PERS0.6.log
#File: exp_sseven_WikipediaNetwork_chameleon_LR0.001_LRE_0.001_WD1e-6_PERS0.8.log

hjt_ablation_experiment() {


	DATA_RUN_PATH="$run_base/$run_path/HJT/PubMed/EXPERIMENTS_HJT"
  mkdir -p "$DATA_RUN_PATH"
	exp_name="exp_${EXPERIMENT}_${MODEL}_${DATASET}_${DATASUBSET}_LR${LR}_LRE_${LREDGE}_WD${WD}_PERS${PERSISTENCE}.log"
  RUN_FILE="$DATA_RUN_PATH/$exp_name"

	PERSISTENCE="0.5,0.7,0.9"
	python main.py --cuda_num=0 \
	--dropout_edge=0.6 --dim_hidden_edge=256 \
	--dropout=0.6 --dim_hidden=256 --dim_gin=256 --dim_multiscale_filter_conv=256 --num_layers=3 \
	--batch_size=64 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
	--dataset=Planetoid --epochs=321 --homophily=0.9 \
	--multi_label=True --type_model=HierGNN \
	--eval_steps=8 --train_by_steps=False \
	--lr=3e-4 --lr2=0.3 --weight_decay=5e-6 \
	--data_subset=PubMed --persistence="${PERSISTENCE}" --hier_model=HJT 2>&1 | tee -- "$RUN_FILE" \
	--experiment=None

	DATA_RUN_PATH="$run_base/$run_path/HJT/Cora/EXPERIMENTS_HJT"
  mkdir -p "$DATA_RUN_PATH"
	exp_name="exp_${EXPERIMENT}_${MODEL}_${DATASET}_${DATASUBSET}_LR${LR}_LRE_${LREDGE}_WD${WD}_PERS${PERSISTENCE}.log"
  RUN_FILE="$DATA_RUN_PATH/$exp_name"
	PERSISTENCE="0.5,0.7,0.9"
	python main.py --cuda_num=0 \
	--dropout_edge=0.6 --dim_hidden_edge=256 \
	--dropout=0.6 --dim_hidden=256 --dim_gin=256 --dim_multiscale_filter_conv=256 --num_layers=3 \
	--batch_size=64 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
	--dataset=Planetoid --epochs=321 --homophily=0.9 \
	--multi_label=True --type_model=HierGNN \
	--eval_steps=8 --train_by_steps=False \
	--lr=0.003 --lr2=0.1 --weight_decay=1e-4 \
	--data_subset=Cora --persistence="${PERSISTENCE}" --hier_model=HJT 2>&1 | tee -- "$RUN_FILE" \
	--experiment=None

  # need tyo transfer runs friom chameleopn, cora , pubmed , and roman to then collect results
  # need to switch study to successive before bed and run both chameleon and cors
}

hjt_ablation_experiment
