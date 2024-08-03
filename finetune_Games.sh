python finetune_data/process_new.py \
    -d Video_Games

python finetune_new.py \
    -d Video_Games \
    --pretrain_ckpt pretrain_ckpt/recformer_ckpt.bin \
    --data_path finetune_data/Video_Games \
    --num_iterations 50 \
    --steps_per_iteration 40000 \
    --batch_size 16 \
    --device 2 \
    --fp16 \
    --finetune_negative_sample_size -1