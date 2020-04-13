import sys

sys.path.append("..")
import os, argparse, random
from tqdm import tqdm, trange
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from vocab import load_vocab
import config as cfg
from bert import model as bert, data
import optimization as optim

torch.cuda.empty_cache()
""" random seed """


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


""" init_process_group """


def init_process_group(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


""" destroy_process_group """


def destroy_process_group():
    dist.destroy_process_group()


""" 모델 epoch 학습 """


def train_epoch(config, rank, epoch, model, criterion_lm, criterion_cls, optimizer, scheduler, train_loader):
    losses = []
    model.train()

    with tqdm(total=len(train_loader), desc=f"Train({rank}) {epoch}") as pbar:
        for i, value in enumerate(train_loader):
            print(type(value))
            print(value)
            labels_cls, labels_lm, inputs, segments = map(lambda v: v.to(config.device), value)
            print(type(labels_cls))
            print(labels_cls)

            optimizer.zero_grad()
            # inputs, segements를 입력으로 BERTPretrain을 실행합니다.
            outputs = model(inputs, segments)
            # 1번의 결과 중 첫 번째 값이 NSP(logits_cls), 두 번째 값이 MLM(logits_lm) 입니다.
            logits_cls, logits_lm = outputs[0], outputs[1]

            # logits_cls 값과 labels_cls 값을 이용해 NSP Loss(loss_cls)를 계산합니다.
            loss_cls = criterion_cls(logits_cls, labels_cls)
            # logits_lm 값과 labels_lm 값을 이용해 MLM Loss(loss_lm)를 계산합니다.
            loss_lm = criterion_lm(logits_lm.view(-1, logits_lm.size(2)), labels_lm.view(-1))
            # loss_cls와 loss_lm을 더해 loss를 생성합니다.
            loss = loss_cls + loss_lm

            loss_val = loss_lm.item()
            losses.append(loss_val)

            # loss, optimizer를 이용해 학습합니다.
            loss.backward()
            optimizer.step()
            scheduler.step()

            pbar.update(1)
            pbar.set_postfix_str(f"Loss: {loss_val:.3f} ({np.mean(losses):.3f})")
    return np.mean(losses)


""" 모델 학습 """


def train_model(rank, world_size, args):
    print('dd22')
    if 1 < args.n_gpu:
        init_process_group(rank, world_size)
    master = (world_size == 0 or rank % world_size == 0)

    vocab = load_vocab(args.vocab)

    config = cfg.Config.load(args.config)
    config.n_enc_vocab = len(vocab)
    # GPU 사용 여부를 확인합니다.
    config.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(config)

    best_epoch, best_loss = 0, 0

    """학습 실행"""
    # BERTPretrain을 생성합니다.
    model = bert.BERTPretrain(config)
    # 기존에 학습된 pretrain 값이 있다면 이를 로드 합니다.
    if os.path.isfile(args.save):
        best_epoch, best_loss = model.bert.load(args.save)
        print(f"rank: {rank} load pretrain from: {args.save}, epoch={best_epoch}, loss={best_loss}")
        best_epoch += 1
    # BERTPretrain이 GPU 또는 CPU를 지원하도록 합니다.
    if 1 < args.n_gpu:
        model.to(config.device)
        model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)
    else:
        model.to(config.device)

    # MLM loss(criterion_lm) 및 NLP loss(criterion_cls) 함수를 선언 합니다.
    criterion_lm = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
    criterion_cls = torch.nn.CrossEntropyLoss()

    train_loader = data.build_pretrain_loader(vocab, args, epoch=best_epoch, shuffle=True)

    t_total = len(train_loader) * args.epoch
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # optimizer를 선언 합니다.
    optimizer = optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = optim.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                      num_training_steps=t_total)

    offset = best_epoch
    losses = []
    for step in trange(args.epoch, desc="Epoch"):
        print('step offset')
        print(step)
        print(offset)
        epoch = step + offset
        # 각 epoch 마다 새로 train_loader를 생성 합니다.
        # step이 0인 경우는 위에서 생성했기 때문에 생성하지 않습니다.
        if 0 < step:
            del train_loader
            train_loader = data.build_pretrain_loader(vocab, args, epoch=epoch, shuffle=True)

        # 각 epoch 마다 학습을 합니다.
        loss = train_epoch(config, rank, epoch, model, criterion_lm, criterion_cls, optimizer, scheduler, train_loader)
        losses.append(loss)

        if master:
            best_epoch, best_loss = epoch, loss
            if isinstance(model, DistributedDataParallel):
                model.module.bert.save(best_epoch, best_loss, args.save)
            else:
                model.bert.save(best_epoch, best_loss, args.save)
            print(f">>>> rank: {rank} save model to {args.save}, epoch={best_epoch}, loss={best_loss:.3f}")

    print(f">>>> rank: {rank} losses: {losses}")
    if 1 < args.n_gpu:
        destroy_process_group()


if __name__ == '__main__':

    print("d")
    import easydict

    args = easydict.EasyDict({
        "config": "config_half.json",
        "vocab": "../kowiki.model",
        "input": "../data/kowiki_bert_{}.json",
        "count": 1,
        "save": "save_pretrain.pth",
        "epoch": 1,
        "batch": 64,
        "gpu": None,
        "seed": 42,
        "weight_decay": 0,
        "learning_rate": 5e-5,
        "adam_epsilon": 1e-8,
        "warmup_steps": 0,
    })

    if torch.cuda.is_available():
        args.n_gpu = torch.cuda.device_count() if args.gpu is None else 1
    else:
        args.n_gpu = 0
    set_seed(args)

    if 1 < args.n_gpu:
        mp.spawn(train_model,
                 args=(args.n_gpu, args),
                 nprocs=args.n_gpu,
                 join=True)
    else:

        train_model(0 if args.gpu is None else args.gpu, args.n_gpu, args)



