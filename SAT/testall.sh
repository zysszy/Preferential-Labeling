python sat.py \
    --logdir=train \
    --gpu_list=1 \
    --train_steps=10000000 \
    --num_threads=16 \
    --save_data_path= \
    --is_evaluate=True \
    --load_model=train/debug-96000 \
    --train_data=train_test_40 \
    --eval_data=satevaldata_newgcn-40 > results/curve.40

python sat.py \
    --logdir=train \
    --gpu_list=1 \
    --train_steps=10000000 \
    --num_threads=16 \
    --save_data_path= \
    --is_evaluate=True \
    --load_model=train/debug-96000 \
    --train_data=train_test_40 \
    --eval_data=satevaldata_newgcn-20 > results/curve.20

python sat.py \
    --logdir=train \
    --gpu_list=1 \
    --train_steps=10000000 \
    --num_threads=16 \
    --save_data_path= \
    --is_evaluate=True \
    --load_model=train/debug-96000 \
    --train_data=train_test_40 \
    --eval_data=satevaldata_newgcn-10 > results/curve.10

python sat.py \
    --logdir=train \
    --gpu_list=1 \
    --train_steps=10000000 \
    --num_threads=16 \
    --save_data_path= \
    --is_evaluate=True \
    --load_model=train/debug-96000 \
    --train_data=train_test_40 \
    --eval_data=satevaldata_newgcn-5 > results/curve.5

