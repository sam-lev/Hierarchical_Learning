#!/bin/sh

#run_folder_path="/home/sam/Documents/PhD/Research/Hierarchical_Learning/Hierarchical_GNN"

run_base=$(pwd)
run_path="runs"

name="$1"

exp_name="exp2_${name}_Wisconsin.log"
exp2() {
python main.py --cuda_num=0 --dropout=0.2 --dim_hidden=512 --num_layers=5 \
--batch_size=32 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
--dataset=WebKB --epochs=80 --homophily=0.9 \
--multi_label=True --type_model=HierGNN \
--eval_steps=50 --train_by_steps=False \
--lr=1e-4 --lr2=1e-4 --weight_decay=1e-8 \
--data_subset=Wisconsin --persistence=0.8 --hier_model=HST 2>&1 | tee -- "$run_base/$run_path/$exp_name"
}
#exp2
exp_name="exp3_${name}_tolokers.log"
exp3() {
python main.py --cuda_num=0 --dropout=0.2 --dim_hidden=512 --num_layers=5 \
--batch_size=256 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
--dataset=HeterophilousGraphDataset --epochs=80 --homophily=0.9 \
--multi_label=False --type_model=HierGNN \
--eval_steps=50 --train_by_steps=False \
--lr=1e-4 --lr2=1e-4 --weight_decay=1e-8 \
--data_subset=tolokers --persistence=0.8 --hier_model=HST 2>&1 | tee -- "$run_base/$run_path/$exp_name"
}
#exp3
exp_name="exp4_${name}_Wisconsin.log"
exp4() {
python main.py --cuda_num=0 --dropout=0.2 --dim_hidden=512 --num_layers=5 \
--batch_size=64 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
--dataset=WebKB --epochs=2 --homophily=0.9 \
--multi_label=True --type_model=HierGNN \
--eval_steps=1 --train_by_steps=False \
--lr=1e-4 --lr2=1e-4 --weight_decay=1e-8 \
--data_subset=wisconsin --persistence=0.3 --hier_model=HJT 2>&1 | tee -- "$run_base/$run_path/$exp_name"
}
exp4
exp_name="exp5_${name}_tolokers.log"
exp5() {
python main.py --cuda_num=0 --dropout=0.2 --dim_hidden=512 --num_layers=5 \
--batch_size=256 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
--dataset=HeterophilousGraphDataset --epochs=80 --homophily=0.9 \
--multi_label=False --type_model=HierGNN \
--eval_steps=50 --train_by_steps=False \
--lr=1e-4 --lr2=1e-4 --weight_decay=1e-12 \
--data_subset=tolokers --persistence=0.8 --hier_model=HST 2>&1 | tee -- "$run_base/$run_path/$exp_name"
}
#exp5
exp_name="exp6_${name}_tolokers.log"
exp6() {
python main.py --cuda_num=0 --dropout=0.2 --dim_hidden=512 --num_layers=5 \
--batch_size=256 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
--dataset=HeterophilousGraphDataset --epochs=80 --homophily=0.9 \
--multi_label=False --type_model=HierGNN \
--eval_steps=50 --train_by_steps=False \
--lr=1e-4 --lr2=1e-4 --weight_decay=0.0 \
--data_subset=tolokers --persistence=0.8 --hier_model=HST 2>&1 | tee -- "$run_base/$run_path/$exp_name"
}
#exp6

#
#/*
#exp_2 () {
#python main.py --cuda_num=0 --dropout=0 --dim_hidden=512 --num_layers=5 \
#--batch_size=128 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
#--dataset=HeterophilousGraphDataset --epochs=1 --homophily=0.9 \
#--multi_label=False --type_model=HierGNN \
#--eval_steps=24 --train_by_steps=True \
#--lr=3e-5 --lr2=1e-2 --weight_decay=1e-5 \
#--data_subset=tolokers --persistence=0.2 --hier_model=HST 2>&1 | tee -- "$run_base/$run_path/$exp_name"
#}
#*/
