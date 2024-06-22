#! /bin/sh -

exp_1 () {
python main.py --cuda_num=0 --dropout=0 --dim_hidden=512 --num_layers=5 \
--batch_size=128 --use_batch_norm=True --SLE_threshold=0.9 --N_exp=1 \
--dataset=HeterophilousGraphDataset --epochs=1 --homophily=0.9 \
--multi_label=False --type_model=HierGNN \
--eval_steps=50 --train_by_steps=True \
--lr=3e-5 --lr2=1e-2 --weight_decay=1e-4 \
--data_subset=tolokers --persistence=0.2,0.5 --hier_model=HST;
} 2>&1 | tee -- "./runs/exp_1.log"




#! /bin/sh -
LOG_FILE="/tmp/abc.log"

{

  # your script here


} 2>&1 | tee -- "$LOG_FILE"
