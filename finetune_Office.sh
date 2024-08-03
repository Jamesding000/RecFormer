python finetune_data/process_new.py \
    -d Office_Products

python finetune_new.py \
    -d Office_Products \
    --pretrain_ckpt pretrain_ckpt/recformer_ckpt.bin \
    --data_path finetune_data/Office_Products \
    --num_iterations 50 \
    --steps_per_iteration 100000 \
    --batch_size 16 \
    --device 3 \
    --fp16 \
    --finetune_negative_sample_size -1