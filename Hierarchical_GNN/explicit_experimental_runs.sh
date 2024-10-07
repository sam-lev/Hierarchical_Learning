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

  MODEL="HST"
  EXPERIMENT="seq_init_ablation"
  DATASET="WikipediaNetwork"
  DATASUBSET="chameleon"
  LR=1e-3
  LREDGE=1e-2
  WD=1e-6
  PERSISTENCE="0.8,0.5,0.3"
	DATA_RUN_PATH="$run_base/$run_path/${MODEL}/EXPERIMENTS/$DATASUBSET"
  mkdir -p "$DATA_RUN_PATH"


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

  # --experiment="$EXPERIMENT" \
  python main.py --cuda_num=0 --experiment="$EXPERIMENT" \
	--dropout_edge=0.0 --dim_hidden_edge=64 \
	--dropout=0.65 --dim_hidden=256 --dim_gin=256 --dim_multiscale_filter_conv=256 --num_layers=3 \
	--batch_size=128 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
	--dataset="$DATASET" --epochs=121 --homophily=0.9 \
	--multi_label=True --type_model=HierGNN \
	--eval_steps=8 --train_by_steps=False \
	--lr=${LR} --lr2=${LREDGE} --weight_decay=${WD} \
	--data_subset="$DATASUBSET" --persistence="${PERSISTENCE}" --hier_model="$MODEL" 2>&1 | tee -- "$RUN_FILE"


  # need tyo transfer runs friom chameleopn, cora , pubmed , and roman to then collect results
  # need to switch study to successive before bed and run both chameleon and cors
}

hjt_ablation_experiment
