python finetune_data/process_new.py \
    -d Video_Games

# --steps_per_iteration should be set to #Users / batch_size to follow the original paper
# total iterations = num_iterations * steps_per_iteration should be set to a fixed value across all datasets
python finetune_new.py \
    -d Video_Games \
    --pretrain_ckpt pretrain_ckpt/recformer_seqrec_ckpt.bin \
    --data_path finetune_data/Video_Games \
    --num_iterations 128 \
    --steps_per_iteration 6000 \
    --batch_size 16 \
    --device 2 \
    --fp16 \
    --finetune_negative_sample_size -1