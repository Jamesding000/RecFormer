python finetune_data/process_new.py \
    -d Cell_Phones_and_Accessories

python finetune_new.py \
    -d Cell_Phones_and_Accessories \
    --pretrain_ckpt pretrain_ckpt/recformer_seqrec_ckpt.bin \
    --data_path finetune_data/Cell_Phones_and_Accessories \
    --num_iterations 128 \
    --steps_per_iteration 10000 \
    --batch_size 12 \
    --device 1 \
    --fp16 \
    --finetune_negative_sample_size -1