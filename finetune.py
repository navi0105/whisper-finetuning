import os
import json
import argparse
import random
import numpy as np
from typing import Iterator, Tuple
from tqdm import tqdm
import copy
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import whisper
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from module.align_model import AlignModel
from dataset import get_dataloader

os.environ["TOKENIZERS_PARALLELISM"]="false"

def parse_args():
    parser = argparse.ArgumentParser()
    # Data Argument
    parser.add_argument(
        '--train-data',
        type=str,
        required=True
    )
    parser.add_argument(
        '--dev-data',
        type=str
    )
    
    # Model Argument
    parser.add_argument(
        '--whisper-model',
        type=str,
        default='large'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='bert-base-chinese'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda'
    )

    # Training Argument
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=4
    )
    parser.add_argument(
        '--dev-batch-size',
        type=int,
        default=16
    )
    parser.add_argument(
        '--accum-grad-steps',
        type=int,
        default=8
    )
    parser.add_argument(
        '--freeze-encoder',
        action='store_true'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-5
    )
    parser.add_argument(
        '--max-grad-norm',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--train-steps',
        type=int,
        default=5000
    )
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=500
    )
    parser.add_argument(
        '--warmup-steps',
        type=int,
        default=500
    )

    parser.add_argument(
        '--save-dir',
        type=str,
        default='result'
    )
    parser.add_argument(
        '--save-all-checkpoints',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=114514
    )


    args = parser.parse_args()
    return args

whisper_dim = {'tiny': 384,
               'base': 512,
               'small': 768,
               'medium': 1024,
               'large': 1280}

def get_phoneme_map():
    phoneme_list = ['blank', 'r', 'ang', 'uo', 'z', 'ai', 'ch', 'i', 'k', 'ou', 'q', 't', 'ian', 'd', 'e', 'j',
                've', 'ong', 'x', 'in', 'iao', 'a', 'ie', 'h', 'ui', 'sh', 'c', 'b', 'ei', 'uan', 'ing', 'm', 'n', 'l'
                , 'u', 'iang', 'zh', 'eng', 'ao', 'en', 'g', 'uang', 'an', 'uai', 'v', 'p', 'f', 'iong', 'uen', 'ua'
                , 'van', 'ia', 'vn', 's', 'o', 'er', 'iu', 'ueng', 'y']
    phoneme_map = {phoneme: idx for idx, phoneme in enumerate(phoneme_list)}
    
    return phoneme_map

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def save_args(args: argparse.Namespace, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(vars(args), indent=4, ensure_ascii=False))

def infinite_iter(data_loader: DataLoader) -> Iterator:
    while True:
        for batch in data_loader:
            yield batch

def train_step(
    model,
    train_iter: Iterator,
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loss_fn,
    accum_grad_steps: int,
    max_grad_norm: float,
) -> Tuple[float, Iterator]:
    model.train()
    total_loss = 0
    total_text_loss = 0
    total_phoneme_loss = 0
    for _ in range(accum_grad_steps):
        mel, y_text, y_phoneme = next(train_iter)
        mel, y_text, y_phoneme = mel.to(model.device), y_text.to(model.device), y_phoneme.to(model.device)

        # Shape: [batch size, time length, number of classes] => (N, T, C)
        output_text, output_phoneme = model(mel)
        text_ctc_loss = compute_ctc_loss(output_text, y_text, loss_fn, model.device)
        phoneme_ctc_loss = compute_ctc_loss(output_phoneme, y_phoneme, loss_fn, model.device)
        
        loss = (text_ctc_loss + phoneme_ctc_loss) / (2 * accum_grad_steps)
        loss.backward()
        total_loss += loss.item()
        total_text_loss += (text_ctc_loss.item() / accum_grad_steps)
        total_phoneme_loss += (phoneme_ctc_loss.item() / accum_grad_steps)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    return (total_loss, total_text_loss, total_phoneme_loss)


@torch.no_grad()
def evaluate(model, dev_loader: DataLoader, loss_fn) -> float:
    model.eval()
    total_loss = 0
    total_text_loss = 0
    total_phoneme_loss = 0
    for mel, y_text, y_phoneme in tqdm(dev_loader):
        mel, y_text, y_phoneme = mel.to(model.device), y_text.to(model.device), y_phoneme.to(model.device)

        # Shape: [batch size, time length, number of classes] => (N, T, C)
        output_text, output_phoneme = model(mel)
        text_ctc_loss = compute_ctc_loss(output_text, y_text, loss_fn, model.device)
        phoneme_ctc_loss = compute_ctc_loss(output_phoneme, y_phoneme, loss_fn, model.device)
        
        loss = text_ctc_loss + phoneme_ctc_loss

        total_loss += loss.item() / 2
        total_text_loss += text_ctc_loss.item()
        total_phoneme_loss += phoneme_ctc_loss.item()

    total_loss /= len(dev_loader)
    total_text_loss /= len(dev_loader)
    total_phoneme_loss /= len(dev_loader)
    return (total_loss, total_text_loss, total_phoneme_loss)

def save_model(model, save_path: str) -> None:
    # save model in half precision to save space
    #model = copy.deepcopy(model).half()
    # save model weights and config in a dictionary that can be loaded with `whisper.load_model`
    torch.save({"model_state_dict": model.state_dict()}, save_path)

def main_loop(
    model,
    train_loader: DataLoader,
    dev_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    loss_fn,
    args: argparse.Namespace,
) -> None:
    min_loss, init_text_loss, init_phoneme_loss = evaluate(model, dev_loader, loss_fn)
    avg_train_loss = 0
    avg_train_text_loss = 0
    avg_train_phoneme_loss = 0
    print(f"Initial loss: {min_loss, init_text_loss, init_phoneme_loss}")
    pbar = tqdm(range(1, args.train_steps + 1))
    train_iter = infinite_iter(train_loader)
    for step in pbar:
        train_loss, train_text_loss, train_phoneme_loss = train_step(
            model,
            train_iter,
            optimizer,
            scheduler,
            loss_fn,
            args.accum_grad_steps,
            args.max_grad_norm,
        )
        pbar.set_postfix({"loss": train_loss, 'text loss:': train_text_loss, 'phoneme loss': train_phoneme_loss})
        avg_train_loss += train_loss
        avg_train_text_loss += train_text_loss
        avg_train_phoneme_loss += train_phoneme_loss
        if step % args.eval_steps == 0:
            eval_loss, eval_text_loss, eval_phoneme_loss = evaluate(model, dev_loader, loss_fn)
            tqdm.write(f"Step {step}: valid loss={eval_loss}, valid text loss={eval_text_loss}, valid phoneme loss={eval_phoneme_loss}")
            tqdm.write(f"Step {step}: train loss={avg_train_loss / args.eval_steps}, \
                        train text loss={avg_train_text_loss / args.eval_steps}, \
                       train phoneme loss={avg_train_phoneme_loss / args.eval_steps}")
            avg_train_loss = 0
        
            if eval_loss < min_loss:
                min_loss = eval_loss
                save_model(model, f"{args.save_dir}/best_model.pt")

            if args.save_all_checkpoints:
                save_model(model, f"{args.save_dir}/step{step}.pt")

            save_model(model, f"{args.save_dir}/last_model.pt")

def compute_ctc_loss(output, labels, ctc_loss, device):
    output_log_sm = F.log_softmax(output, dim=2)

    output_lengths = torch.full(size=(output_log_sm.shape[0], ), fill_value=output_log_sm.shape[1], dtype=torch.long).to(device)
    target_lengths = torch.sum(labels != -100, dim=1)

    loss = ctc_loss(output_log_sm.permute(1, 0, 2), labels, output_lengths, target_lengths)
    return loss

def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    save_args(args, f"{args.save_dir}/args.json")


    device = args.device
    if 'cuda' in device and torch.cuda.is_available() == False:
        device = 'cpu'

    whisper_model = whisper.load_model(args.whisper_model, device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    phoneme_map = get_phoneme_map()

    assert os.path.exists(args.train_data)
    train_dataloader = get_dataloader(data_path=args.train_data,
                                      tokenizer=tokenizer,
                                      phoneme_map=phoneme_map,
                                      batch_size=args.train_batch_size)
    if args.dev_data is not None:
        dev_dataloader = get_dataloader(data_path=args.dev_data,
                                        tokenizer=tokenizer,
                                        phoneme_map=phoneme_map,
                                        batch_size=args.dev_batch_size)

    align_model = AlignModel(whisper_model=whisper_model,
                             embed_dim=whisper_dim[args.whisper_model],
                             text_output_dim=len(tokenizer),
                             phoneme_output_dim=len(phoneme_map),
                             freeze_encoder=args.freeze_encoder,
                             device=device).to(device)
    
    if args.freeze_encoder:
        optimizer = torch.optim.AdamW([{'params': align_model.fc_text.parameters(), 'lr': args.lr},
                                       {'params': align_model.fc_phoneme.parameters(), 'lr': args.lr}],
                                        lr=args.lr,
                                        weight_decay=1e-5
                                  )
    else:
        optimizer = torch.optim.AdamW([{'params': align_model.whisper_model.parameters(), 'lr': args.lr},
                                    {'params': align_model.fc_text.parameters(), 'lr': args.lr},
                                    {'params': align_model.fc_phoneme.parameters(), 'lr': args.lr}],
                                        lr=args.lr,
                                        weight_decay=1e-5
                                    )
    scheduler = scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.train_steps
    )

    ctc_loss = torch.nn.CTCLoss()

    main_loop(
        model=align_model,
        train_loader=train_dataloader,
        dev_loader=dev_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=ctc_loss,
        args=args
    )


if __name__ == "__main__":
    main()