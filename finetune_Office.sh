python finetune_data/process_new.py \
    -d Office_Products

# --steps_per_iteration should be set to #Users / batch_size to follow the original paper
# 223308 / 16 = 14000
# total iterations = num_iterations * steps_per_iteration should be set to a fixed value across all datasets
python finetune_new.py \
    -d Office_Products \
    --pretrain_ckpt pretrain_ckpt/recformer_seqrec_ckpt.bin \
    --data_path finetune_data/Office_Products \
    --num_iterations 55 \
    --steps_per_iteration 14000 \
    --batch_size 16 \
    --device 3 \
    --fp16 \
    --finetune_negative_sample_size -1