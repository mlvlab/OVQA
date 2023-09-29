import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import math
import sys
from typing import Iterable
import argparse
import time
import datetime
from util import dist
from torch.utils.data import DataLoader, DistributedSampler
from collections import namedtuple
from functools import reduce
import pickle

from dataset import VideoQA_Dataset, videoqa_collate_fn
from args import get_args_parser
from util.misc import get_mask, adjust_learning_rate
from util.metrics import MetricLogger

from transformers import DebertaV2Tokenizer
from model import DebertaV2ForMaskedLM

def train_one_epoch(model: torch.nn.Module, tokenizer, data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int,
                    dataset_name, args, max_norm: float = 0):
    model.train()
    edge_index = data_loader.dataset.edge_index.to(device)
    vocab_embeddings = data_loader.dataset.vocab_embeddings.to(device)
    eps = data_loader.dataset.eps[:, None].to(device)

    metric_logger = MetricLogger(delimiter="  ")
    header = "Epoch: [{}]".format(epoch)
    num_training_steps = int(len(data_loader) * args.epochs)
    args.print_freq = int(len(data_loader) / 4)
    for i_batch, batch_dict in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        video = batch_dict["video"].to(device)
        video_len = batch_dict["video_len"]
        video_mask = get_mask(video_len, video.size(1)).to(device)
        text = batch_dict["text"]
        encoded = tokenizer(text, add_special_tokens=True, max_length=args.max_tokens, padding="longest", truncation=True, return_tensors="pt")

        inputs = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        # forward
        answer_id = batch_dict["answer_id"].to(device)
        output = model(video=video, video_mask=video_mask, input_ids=inputs, attention_mask=attention_mask, edge_index=edge_index, vocab_embeddings=vocab_embeddings, eps=eps)
        delay = args.max_feats if args.use_video else 0
        logits = output['logits']
        logits = logits[:, delay:encoded["input_ids"].size(1) + delay][encoded["input_ids"] == tokenizer.mask_token_id]
        
        if dataset_name == "ivqa":
            a = (answer_id / 2).clamp(max=1)
            nll = -F.log_softmax(logits, 1, _stacklevel=5)
            loss = (nll * a / a.sum(1, keepdim=True).clamp(min=1)).sum(dim=1).mean()
        elif dataset_name == "vqa":
            a = (answer_id / 3).clamp(max=1)
            nll = -F.log_softmax(logits, 1, _stacklevel=5)
            loss = (nll * a / a.sum(1, keepdim=True).clamp(min=1)).sum(dim=1).mean()
        else:   
            loss = F.cross_entropy(logits, answer_id)

        loss_dict = {"cls_loss": loss}

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = dist.reduce_dict(loss_dict)
        loss_reduced = sum(loss_dict_reduced.values())
        loss_value = loss_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        adjust_learning_rate(optimizer, curr_step=epoch * len(data_loader) + i_batch, num_training_steps=num_training_steps, args=args)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model: torch.nn.Module, tokenizer, data_loader, device: torch.device, dataset_name, args, thresholds=[1, 10], split="test", epoch=-1):
    model.eval()
    ans2cat = data_loader.dataset.ans2cat
    class_tensor = torch.zeros((len(data_loader.dataset.ans2id), 2), dtype=torch.float64, device="cuda")
    edge_index = data_loader.dataset.edge_index.to(device)
    vocab_embeddings = data_loader.dataset.vocab_embeddings.to(device)
    eps = data_loader.dataset.eps[:, None].to(device)

    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.update(n=0, base=0)
    metric_logger.update(n=0, common=0)
    metric_logger.update(n=0, rare=0)
    metric_logger.update(n=0, unseen=0)
    metric_logger.update(n=0, total=0)
    header = f"{split}:"
    
    args.print_freq = int(len(data_loader) / 4)
    for i_batch, batch_dict in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        video = batch_dict["video"].to(device)
        video_len = batch_dict["video_len"]
        video_mask = get_mask(video_len, video.size(1)).to(device)
        text = batch_dict["text"]
        encoded = tokenizer(text, add_special_tokens=True, max_length=args.max_tokens, padding="longest", truncation=True, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if not args.suffix and not args.use_context:  # remove sep token if not using the suffix
            attention_mask[input_ids == tokenizer.sep_token_id] = 0
            input_ids[input_ids == tokenizer.sep_token_id] = tokenizer.pad_token_id

    
        answer_id, qids = batch_dict["answer_id"].to(device), batch_dict["qid"]
        output = model(video=video, video_mask=video_mask, input_ids=input_ids, attention_mask=attention_mask, edge_index=edge_index, vocab_embeddings=vocab_embeddings, eps=eps)
        logits = output["logits"]
        delay = args.max_feats if args.use_video else 0
        logits = logits[:, delay:encoded["input_ids"].size(1) + delay][encoded["input_ids"] == tokenizer.mask_token_id]  # get the prediction on the mask token
        logits = logits.softmax(-1)
        
        topk_logits, topk_aids = torch.topk(logits, max(thresholds), -1)

        types = batch_dict["type"]
        original_answers = batch_dict['original_answer']
        
        
        for i, (p, ans) in enumerate(zip(answer_id == logits.max(1).indices, original_answers)):
            category = ans2cat[ans]
            class_tensor[answer_id[i]][0] += p.float().item()
            class_tensor[answer_id[i]][1] += 1.
            if category == 'base':
                metric_logger.update(n=1, base=p.float().item())
            elif category == 'common':
                metric_logger.update(n=1, common=p.float().item())
            elif category == 'rare':
                metric_logger.update(n=1, rare=p.float().item())
            elif category == 'unseen':
                metric_logger.update(n=1, unseen=p.float().item())
            metric_logger.update(n=1, total=p.float().item())
    

    torch.distributed.barrier()
    torch.distributed.all_reduce(class_tensor)
    macc = (class_tensor[:, 0] / class_tensor[:, 1]).mean().item()
    metric_logger.synchronize_between_processes()
    metric_logger.update(n=1, macc=macc)
    print("Averaged stats:", metric_logger)
    
    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return results


def main(args):
    # Init distributed mode
    dist.init_distributed_mode(args)
    if dist.is_main_process():
        if args.save_dir and not (os.path.isdir(args.save_dir)):
            os.makedirs(os.path.join(args.save_dir), exist_ok=True)
        print(args)

    device = torch.device(args.device)

    # Fix seeds
    seed = args.seed + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Build model
    tokenizer = DebertaV2Tokenizer.from_pretrained(args.model_name, local_files_only=True)
    
    dataset_test = VideoQA_Dataset(args, tokenizer, "test")
    sampler_test = DistributedSampler(dataset_test, shuffle=False) if args.distributed else torch.utils.data.SequentialSampler(dataset_test)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size_test, sampler=sampler_test, collate_fn=videoqa_collate_fn, num_workers=args.num_workers)
                
    if not args.eval:
        dataset_train = VideoQA_Dataset(args, tokenizer, 'train')
        sampler_train = DistributedSampler(dataset_train) if args.distributed else torch.utils.data.RandomSampler(dataset_train)
        dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler_train, collate_fn=videoqa_collate_fn, num_workers=args.num_workers)
    
    args.n_ans = len(dataloader_test.dataset.ans2id)
    
    model = DebertaV2ForMaskedLM.from_pretrained(features_dim=args.features_dim if args.use_video else 0, max_feats=args.max_feats, freeze_lm=args.freeze_lm, 
                                                 freeze_mlm=args.freeze_mlm, ft_ln=args.ft_ln, ds_factor_attn=args.ds_factor_attn, ds_factor_ff=args.ds_factor_ff, 
                                                 dropout=args.dropout, n_ans=args.n_ans, freeze_last=args.freeze_last, pretrained_model_name_or_path=args.model_name, 
                                                 local_files_only=True, args=args)
    model.to(device)
    
    total_parameters = sum(p.numel() for p in model.parameters())
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total params: {total_parameters:,}')
    print(f'Trained params: {n_parameters:,}')

    # Set up optimizer
    params_for_optimization = list(p for n, p in model.named_parameters() if (p.requires_grad and 'gat' not in n))
    answer_params_for_optimization = list(p for n, p in model.named_parameters() if (p.requires_grad and 'gat' in n))
    optimizer = torch.optim.Adam([{"params": params_for_optimization, "lr": args.lr}, {"params": answer_params_for_optimization, "lr": args.lr}],
                                 lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay)
    
    # Load pretrained checkpoint
    if args.load:
        print("loading from", args.load)
        checkpoint = torch.load(args.load, map_location="cpu")
        if 'model' in checkpoint.keys():
            model.load_state_dict(checkpoint['model'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        
        if args.resume and not args.eval:
            optimizer.load_state_dict(checkpoint["optimizer"])
            args.start_epoch = checkpoint["epoch"] + 1

    if not args.eval:
        train_aid2tokid = torch.zeros(len(dataloader_train.dataset.ans2id), args.max_atokens).long()
        for a, aid in dataloader_train.dataset.ans2id.items():
            tok = torch.tensor(tokenizer(a, add_special_tokens=False, max_length=args.max_atokens, truncation=True, padding="max_length")["input_ids"], dtype=torch.long)
            train_aid2tokid[aid] = tok
        
        print(f'Training Vocab : {len(train_aid2tokid)}')
        print(f'Training Samples : {len(dataloader_train.dataset)}')
    test_aid2tokid = torch.zeros(len(dataloader_test.dataset.ans2id), args.max_atokens).long()
    for a, aid in dataloader_test.dataset.ans2id.items():
        tok = torch.tensor(tokenizer(a, add_special_tokens=False, max_length=args.max_atokens, truncation=True, padding="max_length")["input_ids"], dtype=torch.long)
        test_aid2tokid[aid] = tok
    print(f'Test Vocab : {len(test_aid2tokid)}')
    print(f'Test Samples : {len(dataloader_test.dataset)}')

    if not args.eval:
        print("Start training")
        start_time = time.time()
        best_epoch = args.start_epoch
        best_acc = 0
        for epoch in range(args.start_epoch, args.epochs):
            print(f"Starting epoch {epoch}")
            if args.distributed:
                sampler_train.set_epoch(epoch)
            
            model.set_answer_embeddings(train_aid2tokid.to(model.device), freeze_last=args.freeze_last)
            train_stats = train_one_epoch(model=model, tokenizer=tokenizer, data_loader=dataloader_train, optimizer=optimizer, device=device, epoch=epoch,
                                          dataset_name=args.dataset, args=args, max_norm=args.clip_max_norm)

            if (epoch + 1) % args.eval_skip == 0:
                print(f"Validating {args.dataset}")
                val_stats = {}
                model.set_answer_embeddings(test_aid2tokid.to(model.device), freeze_last=args.freeze_last)
                results = evaluate(model=model, tokenizer=tokenizer, data_loader=dataloader_test, device=device, dataset_name=args.dataset, 
                                   args=args, split="val", epoch=epoch)
                val_stats.update({args.dataset + "_" + k: v for k, v in results.items()})

                if results["total"] > best_acc:
                    best_epoch = epoch
                    best_acc = results["total"]
                    if dist.is_main_process() and args.save_dir:
                        checkpoint_path = os.path.join(args.save_dir, f"best_model.pth")
                        dist.save_on_master({"model": model, "optimizer": optimizer.state_dict(), "epoch": epoch, "args": args}, checkpoint_path)
                        json.dump({"acc": best_acc, "ep": epoch}, open(os.path.join(args.save_dir, args.dataset + "acc_val.json"), "w"))
            else:
                val_stats = {}

            log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, **{f"val_{k}": v for k, v in val_stats.items()}, "epoch": epoch, "n_parameters": n_parameters}

            if args.save_dir and dist.is_main_process():
                with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                checkpoint_path = os.path.join(args.save_dir, f"ckpt.pth")
                dist.save_on_master({"model": model, "optimizer": optimizer.state_dict(), "epoch": epoch, "args": args}, checkpoint_path)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))
        # load best ckpt
        if dist.is_main_process() and args.save_dir:
            print(f"loading best checkpoint from epoch {best_epoch}")
        if args.save_dir:
            torch.distributed.barrier()  # wait all processes
            checkpoint = torch.load(os.path.join(args.save_dir, f"best_model.pth"), map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)

    model.set_answer_embeddings(test_aid2tokid.to(model.device), freeze_last=args.freeze_last)
    results = evaluate(model=model, tokenizer=tokenizer, data_loader=dataloader_test, device=device, dataset_name=args.dataset,
                       args=args, split="val" if (args.eval and not args.test) else "test")
                
    if args.save_dir and dist.is_main_process():
        json.dump(results, open(os.path.join(args.save_dir, args.dataset + ".json"), "w"))
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args()

    if args.save_dir:
        args.save_dir = os.path.join(args.presave_dir, args.save_dir)
    args.model_name = os.path.join('./pretrained', args.model_name)
    main(args)
