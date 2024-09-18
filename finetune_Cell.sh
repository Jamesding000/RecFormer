python finetune_data/process_new.py \
    -d Cell_Phones_and_Accessories

# --steps_per_iteration should be set to #Users / batch_size to follow the original paper
# 381000 / 12 = 30000
# total iterations = num_iterations * steps_per_iteration should be set to a fixed value across all datasets
python finetune_new.py \
    -d Cell_Phones_and_Accessories \
    --pretrain_ckpt pretrain_ckpt/recformer_seqrec_ckpt.bin \
    --data_path finetune_data/Cell_Phones_and_Accessories \
    --num_iterations 30 \
    --steps_per_iteration 30000 \
    --batch_size 12 \
    --device 1 \
    --fp16 \
    --finetune_negative_sample_size -1