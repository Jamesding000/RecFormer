python finetune.py \
    --pretrain_ckpt pretrain_ckpt/recformer_ckpt.bin \
    --data_path finetune_data/Scientific \
    --num_train_epochs 128 \
    --batch_size 16 \
    --device 5 \
    --fp16 \
    --finetune_negative_sample_size -1