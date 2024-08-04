import os
import json
import torch
from tqdm import tqdm
from multiprocessing import Pool
from pathlib import Path
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from pytorch_lightning import seed_everything

from utils import read_json, AverageMeterSet, Ranker
from optimization import create_optimizer_and_scheduler
from recformer import RecformerModel, RecformerForSeqRec, RecformerTokenizer, RecformerConfig
from collator_new import FinetuneDataCollatorWithPadding, EvalDataCollatorWithPadding
# from dataloader import RecformerTrainDataset, RecformerEvalDataset

from specGR_utils import load_dataset_splits

def prepare_recformer_datasets(input_sequence, label, max_length):
    input_ids = [int(i)-1 for i in input_sequence]
    return {
        'items': input_ids,
        'label': label-1
    } # Recformer Uses 0-based indexing

def load_data(args, splits, max_history_len, num_workers=1):
    
    datasets = load_dataset_splits(args.d, splits)
    datas = {}
    
    for split in splits:
        max_inputs_length = max_history_len
        result_dataset = datasets[split].map(
            lambda t: prepare_recformer_datasets(list(map(int, t['history'].split(' '))), t['item_id'], max_inputs_length),
            num_proc=num_workers
        )
        result_dataset = result_dataset.select_columns(['items', 'label'])
        # result_dataset.set_format(type="list")
    
        datas[split] = result_dataset
    
    item_meta_dict = json.load(open(os.path.join(args.data_path, args.meta_file)))
    item2id = read_json(os.path.join(args.data_path, args.item2id_file))
    id2item = {v:k for k, v in item2id.items()}
    
    item_meta_dict_filted = dict()
    for k, v in item_meta_dict.items():
        if k in item2id:
            item_meta_dict_filted[k] = v
    
    return datas, item_meta_dict_filted, item2id, id2item


tokenizer_glb: RecformerTokenizer = None
def _par_tokenize_doc(doc):
    
    item_id, item_attr = doc

    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids, token_type_ids

def encode_all_items(model: RecformerModel, tokenizer: RecformerTokenizer, tokenized_items, args):

    model.eval()

    items = sorted(list(tokenized_items.items()), key=lambda x: x[0])
    items = [ele[1] for ele in items]

    item_embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(items), args.batch_size), ncols=100, desc='Encode all items'):

            item_batch = [[item] for item in items[i:i+args.batch_size]]

            inputs = tokenizer.batch_encode(item_batch, encode_item=False)

            for k, v in inputs.items():
                inputs[k] = torch.LongTensor(v).to(args.device)

            outputs = model(**inputs)

            item_embeddings.append(outputs.pooler_output.detach())

    item_embeddings = torch.cat(item_embeddings, dim=0)#.cpu()

    return item_embeddings


def eval(model, dataloader, args):
    model.eval()

    ranker = Ranker(args.metric_ks)
    average_meter_set = AverageMeterSet()

    max_k = max(args.metric_ks)
    recall_key = f"Recall@{max_k}"
    ndcg_key = f"NDCG@{max_k}"

    # Initialize the progress bar
    pbar = tqdm(dataloader, ncols=100, desc='Evaluate', unit='batch')

    for batch, labels in pbar:
        for k, v in batch.items():
            batch[k] = v.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            scores = model(**batch)

        res = ranker(scores, labels)

        metrics = {}
        for i, k in enumerate(args.metric_ks):
            metrics[f"NDCG@{k}"] = res[2*i]
            metrics[f"Recall@{k}"] = res[2*i+1]
        metrics["MRR"] = res[-3]
        metrics["AUC"] = res[-2]

        for k, v in metrics.items():
            average_meter_set.update(k, v)

        # Update the progress bar with Recall@max_k and NDCG@max_k
        averages = average_meter_set.averages()
        average_recall = averages[recall_key]
        average_ndcg = averages[ndcg_key]
        pbar.set_postfix({recall_key: f'{average_recall:.4f}', ndcg_key: f'{average_ndcg:.4f}'})

    average_metrics = average_meter_set.averages()

    return average_metrics


# def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, args):

#     model.train()

#     for step, batch in enumerate(tqdm(dataloader, ncols=100, desc='Training')):
#         for k, v in batch.items():
#             batch[k] = v.to(args.device)

#         if args.fp16:
#             with autocast():
#                 loss = model(**batch)
#         else:
#             loss = model(**batch)

#         if args.gradient_accumulation_steps > 1:
#             loss = loss / args.gradient_accumulation_steps

#         if args.fp16:
#             scaler.scale(loss).backward()
#         else:
#             loss.backward()

#         if (step + 1) % args.gradient_accumulation_steps == 0:
#             if args.fp16:

#                 scale_before = scaler.get_scale()
#                 scaler.step(optimizer)
#                 scaler.update()
#                 scale_after = scaler.get_scale()
#                 optimizer_was_run = scale_before <= scale_after
#                 optimizer.zero_grad()

#                 if optimizer_was_run:
#                     scheduler.step()

#             else:

#                 scheduler.step()  # Update learning rate schedule
#                 optimizer.step()
#                 optimizer.zero_grad()

from tqdm import tqdm

def train_one_iteration(model, dataloader, optimizer, scheduler, scaler, args, steps, train_dataset, finetune_data_collator):
    model.train()
    total_loss = 0
    
    # Initialize tqdm progress bar
    progress_bar = tqdm(range(steps), ncols=100, desc='Training')

    # Convert dataloader to an iterator
    data_iter = iter(dataloader)
    
    for step in progress_bar:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reinitialize iterator if we've reached the end of the dataset
            dataloader = DataLoader(train_dataset, 
                        batch_size=args.batch_size, 
                        shuffle=True, 
                        collate_fn=finetune_data_collator)
            data_iter = iter(dataloader)
            batch = next(data_iter)

        for k, v in batch.items():
            batch[k] = v.to(args.device)
        
        if args.fp16:
            with autocast():
                loss = model(**batch)
        else:
            loss = model(**batch)
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        if args.fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.item()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            if args.fp16:
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                scale_after = scaler.get_scale()
                optimizer_was_run = scale_before <= scale_after
                optimizer.zero_grad()
                if optimizer_was_run:
                    scheduler.step()
            else:
                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                optimizer.zero_grad()
        
        # Update progress bar with current loss
        progress_bar.set_postfix({'loss': total_loss / (step + 1)})
    
    return data_iter



def main():
    parser = ArgumentParser()
    # path and file
    parser.add_argument('-d', type=str, default=None, required=True)
    parser.add_argument('--pretrain_ckpt', type=str, default=None, required=True)
    parser.add_argument('--data_path', type=str, default=None, required=True)
    parser.add_argument('--output_dir', type=str, default='checkpoints')
    parser.add_argument('--ckpt', type=str, default='best_model.bin')
    parser.add_argument('--model_name_or_path', type=str, default='allenai/longformer-base-4096')
    parser.add_argument('--train_file', type=str, default='train.json')
    parser.add_argument('--dev_file', type=str, default='val.json')
    parser.add_argument('--test_file', type=str, default='test.json')
    parser.add_argument('--item2id_file', type=str, default='smap.json')
    parser.add_argument('--meta_file', type=str, default='meta_data.json')

    # data process
    parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
    parser.add_argument('--dataloader_num_workers', type=int, default=0)

    # model
    parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")

    # train
    parser.add_argument('--steps_per_iteration', type=int, default=100000)
    parser.add_argument('--num_iterations', type=int, default=16)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
    parser.add_argument('--finetune_negative_sample_size', type=int, default=1000)
    parser.add_argument('--metric_ks', nargs='+', type=int, default=[10, 50], help='ks for Metric@k')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--warmup_steps', type=int, default=100)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--fix_word_embedding', action='store_true')
    parser.add_argument('--verbose', type=int, default=3)
    

    args = parser.parse_args()
    print(args)
    seed_everything(42)
    args.device = torch.device('cuda:{}'.format(args.device)) if args.device>=0 else torch.device('cpu')
    
    splits = ['train', 'valid', 'test','test_in_sample', 'test_cold_start']
    datasets, item_meta_dict, item2id, id2item = load_data(args, splits, 20, num_workers=4)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * 12
    config.max_token_num = 256 #1024
    config.item_num = len(item2id)
    config.finetune_negative_sample_size = args.finetune_negative_sample_size
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)
    
    global tokenizer_glb
    tokenizer_glb = tokenizer

    path_corpus = Path(args.data_path)
    dir_preprocess = path_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_output = Path(args.output_dir) / path_corpus.name
    path_output.mkdir(exist_ok=True, parents=True)
    path_ckpt = path_output / args.ckpt

    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'

    # if path_tokenized_items.exists():
    #     print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    # else:
    print(f'Loading attribute data {path_corpus}')
    pool = Pool(processes=args.preprocessing_num_workers)
    pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_meta_dict.items())
    doc_tuples = list(tqdm(pool_func, total=len(item_meta_dict), ncols=100, desc=f'[Tokenize] {path_corpus}'))
    tokenized_items = {item2id[item_id]: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
    pool.close()
    pool.join()

    torch.save(tokenized_items, path_tokenized_items)

    tokenized_items = torch.load(path_tokenized_items)
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    finetune_data_collator = FinetuneDataCollatorWithPadding(tokenizer, tokenized_items)
    eval_data_collator = EvalDataCollatorWithPadding(tokenizer, tokenized_items)
    # train_data = RecformerTrainDataset(train, collator=finetune_data_collator)
    # val_data = RecformerEvalDataset(train, val, test, mode='val', collator=eval_data_collator)
    # test_data = RecformerEvalDataset(train, val, test, mode='test', collator=eval_data_collator)
    
    # for i in range(10):
    #     print(f'****** {i} ******')
    #     print('train', train_data[0])
    #     print('val', val_data[0])
    
    # import time
    # time.sleep(10)
    
    # train_loader = DataLoader(train_data, 
    #                           batch_size=args.batch_size, 
    #                           shuffle=True, 
    #                           collate_fn=train_data.collate_fn)
    # dev_loader = DataLoader(val_data, 
    #                         batch_size=args.batch_size, 
    #                         collate_fn=val_data.collate_fn)
    # test_loader = DataLoader(test_data, 
    #                         batch_size=args.batch_size, 
    #                         collate_fn=test_data.collate_fn)
    
    train_loader = DataLoader(datasets['train'], 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            collate_fn=finetune_data_collator)
    dev_loader = DataLoader(datasets['valid'], 
                            batch_size=args.batch_size, 
                            collate_fn=eval_data_collator)
    # test_loader = DataLoader(datasets['test'], 
    #                         batch_size=args.batch_size, 
    #                         collate_fn=eval_data_collator)
    test_in_sample_loader = DataLoader(datasets['test_in_sample'], 
                        batch_size=args.batch_size, 
                        collate_fn=eval_data_collator)
    test_cold_start_loader = DataLoader(datasets['test_cold_start'], 
                        batch_size=args.batch_size, 
                        collate_fn=eval_data_collator)

    model = RecformerForSeqRec(config)
    pretrain_ckpt = torch.load(args.pretrain_ckpt)
    model.load_state_dict(pretrain_ckpt, strict=False)
    model.to(args.device)

    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    path_item_embeddings = dir_preprocess / f'item_embeddings_{path_corpus.name}'
    if path_item_embeddings.exists():
        print(f'[Item Embeddings] Use cache: {path_tokenized_items}')
    else:
        print(f'Encoding items.')
        item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
        torch.save(item_embeddings, path_item_embeddings)
    
    item_embeddings = torch.load(path_item_embeddings)
    model.init_item_embedding(item_embeddings)

    model.to(args.device) # send item embeddings to device

    num_train_optimization_steps = args.num_iterations * args.steps_per_iteration // args.gradient_accumulation_steps
    optimizer, scheduler = create_optimizer_and_scheduler(model, num_train_optimization_steps, args)

    if args.fp16:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # test_metrics = eval(model, test_loader, args)
    # print(f'Test set: {test_metrics}')
    
    best_target = float('-inf')
    patient = 5

    train_iter = iter(train_loader)

    for iteration in range(args.num_iterations):
        if iteration == 0 or not train_iter:
            item_embeddings = encode_all_items(model.longformer, tokenizer, tokenized_items, args)
            model.init_item_embedding(item_embeddings)
            train_iter = iter(train_loader)

        train_iter = train_one_iteration(model, train_iter, optimizer, scheduler, scaler, args, args.steps_per_iteration, datasets['train'], finetune_data_collator)
        
        if (iteration + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args)
            print(f'Iteration: {iteration}. Dev set: {dev_metrics}')

            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 5
                torch.save(model.state_dict(), path_ckpt)
            else:
                patient -= 1
                if patient == 0:
                    break

    print('Load best model in stage 1.')
    model.load_state_dict(torch.load(path_ckpt))
        
    patient = 3
    train_iter = iter(train_loader)

    for iteration in range(args.num_iterations):
        if not train_iter:
            train_iter = iter(train_loader)

        train_iter = train_one_iteration(model, train_iter, optimizer, scheduler, scaler, args, args.steps_per_iteration, datasets['train'], finetune_data_collator)
        
        if (iteration + 1) % args.verbose == 0:
            dev_metrics = eval(model, dev_loader, args)
            print(f'Iteration: {iteration}. Dev set: {dev_metrics}')

            if dev_metrics['NDCG@10'] > best_target:
                print('Save the best model.')
                best_target = dev_metrics['NDCG@10']
                patient = 3
                torch.save(model.state_dict(), path_ckpt)
            else:
                patient -= 1
                if patient == 0:
                    break


    def log_metrics(pretrain_ckpt, data_path, in_sample_metrics, cold_start_metrics):
        # Create logs directory if it doesn't exist
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)

        # Create a unique file name based on pretrain_ckpt and data_path
        file_name = f"metrics_{os.path.basename(pretrain_ckpt)}_{os.path.basename(data_path)}.txt"
        file_path = os.path.join(logs_dir, file_name)

        # Write metrics to the file
        with open(file_path, "w") as f:
            f.write("Test set (In-sample):\n")
            for key, value in in_sample_metrics.items():
                f.write(f"{key}: {value}\n")

            f.write("\nTest set (Cold-start):\n")
            for key, value in cold_start_metrics.items():
                f.write(f"{key}: {value}\n")

        print(f"Metrics logged to {file_path}")

    # Example usage
    print('Test with the best checkpoint.')  
    model.load_state_dict(torch.load(path_ckpt))
    test_in_sample_metrics = eval(model, test_in_sample_loader, args)
    print(f'Test set: {test_in_sample_metrics}')
        
    test_cold_start_metrics = eval(model, test_cold_start_loader, args)
    print(f'Test set: {test_cold_start_metrics}')

    # Log the metrics
    log_metrics(args.pretrain_ckpt, args.data_path, test_in_sample_metrics, test_cold_start_metrics)
                
if __name__ == "__main__":
    main()