import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List, Dict
import numpy as np
import math
import logging

from tqdm.auto import tqdm
import torch
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_scheduler,
    set_seed,
    SchedulerType,
)

from dataset import SeqClsDataset
from model import SeqClsModel


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

TRAIN = "train"
DEV = "eval"
TEST = "test"
SPLITS = [TRAIN, DEV, TEST]

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_file",
        type=Path,
        help="Directory to the input data.",
        required=True,
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to store the model parameters.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches and label mapping.",
        required=True,
    )
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--seed", type=int, default=13, help="A seed for reproducible training.")
    parser.add_argument(
        "--device", 
        type=torch.device,
        default="cuda",
        help="cpu, cuda, cuda:0, cuda:1"
    )

    args = parser.parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    # load input data, default json file. 
    # If data file is not json, please rewrite this section.
    data = json.loads(args.data_file.read_text())

    # load label mapping, default stored in cache directory.
    # If label mapping does not exist, please write a section to get it.
    label_mapping_path = args.cache_dir / "label_mapping.json"
    label_mapping = json.loads(label_mapping_path.read_text())
    
    # Build datasets and dataloaders
    datasets = {split: SeqClsDataset(data[split], label_mapping, args.max_len, args.model_name_or_path) for split in SPLITS}
    dataloaders = {
        split: DataLoader(
            datasets[split], 
            batch_size=args.per_device_train_batch_size, 
            shuffle=True, 
            collate_fn=datasets[split].collate_fn)
        if split == TRAIN else
        split: DataLoader(
            datasets[split], 
            batch_size=args.per_device_eval_batch_size, 
            shuffle=False, 
            collate_fn=datasets[split].collate_fn)
        for split in SPLITS
    }

    # Intialize model
    model = SeqClsModel(args.model_name_or_path, datasets[TRAIN].num_classes)              
    model.to(device=args.device)
        
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )
        
    # Train!
    total_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps

    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {len(train_dataset)}")
    logging.info(f"  Num Epochs = {args.num_train_epochs}")
    logging.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logging.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logging.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logging.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0
    best_eval_score = 0
    ckpt_path = args.ckpt_dir / "model_weights.pth"

    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(dataloader[TRAIN]):
            batch = {key: torch.tensor(value, device=args.device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.set_description(f"Epoch {epoch}")
                progress_bar.set_postfix(loss=loss.item())
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break
        
        # evaluation
        model.eval()
        score = 0
        for step, batch in enumerate(dataloader[DEV]):
            batch = {key: torch.tensor(value, device=args.device) for key, value in batch.items()}
            outputs = model(**batch)
            labels = batch["labels"].cpu().detach().numpy()
            predictions = outputs.logits.cpu().detach().numpy().argmax(-1)
            score += (labels==predictions).sum()
        
        score /= len(dataloader[DEV])
        if best_score < score:
            best_score = score
            torch.save(model.state_dict(), ckpt_path)
        print(f"Epoch {epoch} score: {score}")

    # test
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    expected, predicted = [], []
    for step, batch in enumerate(dataloader[TEST]):
        batch = {key: torch.tensor(value, device=args.device) for key, value in batch.items()}
        outputs = model(**batch)
        labels = batch["labels"].cpu().detach().numpy().tolist()
        predictions = outputs.logits.cpu().detach().numpy().argmax(-1).tolist()
        expected.extend(labels)
        predicted.extend(predictions)

    save_path = args.cache_dir / "output.json"
    save_path.write_text(json.dumps({"expected": expected, "predicted": predicted}, indent=2))
    logging.info(f"Results saved at {str(save_path.resolve())}")


if __name__ == "__main__":
    main()