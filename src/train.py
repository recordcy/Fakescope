"""
train.py — 4차 시도: Midjourney + CSV 데이터셋 추가
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(__file__))

from dataset import get_dataloaders
from model import FakeDetector


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for i, (imgs, labels) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs          = imgs.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs

        if (i + 1) % 100 == 0:
            print(f"  [epoch {epoch}] step {i+1}/{len(loader)} | "
                  f"loss={loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)

        bs          = imgs.size(0)
        total_loss += loss.item() * bs
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += bs

    return total_loss / total, correct / total


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")

    loaders = get_dataloaders(
        faces_dir   = args.faces_dir,
        pixiv_dir   = args.pixiv_dir,
        intel_dir   = args.intel_dir,
        ai_gen_dir  = args.ai_gen_dir,
        coco_dir    = args.coco_dir,
        mj_dir      = args.mj_dir,
        csv_path    = args.csv_path,
        csv_img_dir = args.csv_img_dir,
        batch_size  = args.batch_size,
        num_workers = 2,
        coco_sample = args.coco_sample,
    )

    model     = FakeDetector(use_freq=args.use_fft).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    drive_dir = '/content/drive/MyDrive/fakescope_checkpoints'
    os.makedirs(args.weights_dir, exist_ok=True)
    use_drive = os.path.isdir('/content/drive/MyDrive')
    ckpt_path = os.path.join(
        drive_dir if use_drive else args.weights_dir, 'best_model.pth')
    if use_drive:
        os.makedirs(drive_dir, exist_ok=True)

    start_epoch  = 1
    best_val_acc = 0.0
    patience_cnt = 0

    if os.path.exists(ckpt_path):
        print("이전 체크포인트 발견 → 이어서 학습")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        scheduler.load_state_dict(ckpt['sched_state'])
        start_epoch  = ckpt['epoch'] + 1
        best_val_acc = ckpt['val_acc']
        print(f"  재개 에폭: {start_epoch}, 이전 best: {best_val_acc:.4f}")

    print(f"\n{'='*55}")
    print(f"학습 시작: {start_epoch} ~ {args.epochs} 에폭")
    print(f"{'='*55}")

    history = {'train_loss': [], 'train_acc': [],
               'val_loss':   [], 'val_acc':   []}

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one_epoch(
            model, loaders['train'], criterion, optimizer, device, epoch)
        vl_loss, vl_acc = evaluate(
            model, loaders['valid'], criterion, device)

        scheduler.step()
        elapsed = time.time() - t0

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(vl_loss)
        history['val_acc'].append(vl_acc)

        print(f"\nEpoch {epoch:3d}/{args.epochs} ({elapsed:.0f}s) | "
              f"lr={scheduler.get_last_lr()[0]:.2e}")
        print(f"  Train loss={tr_loss:.4f}  acc={tr_acc:.4f}")
        print(f"  Val   loss={vl_loss:.4f}  acc={vl_acc:.4f}", end='')

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            patience_cnt = 0
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'sched_state': scheduler.state_dict(),
                'val_acc'    : vl_acc,
                'use_freq'   : args.use_fft,
            }, ckpt_path)
            print(f"  ✅ 저장! (best={best_val_acc:.4f})")
        else:
            patience_cnt += 1
            print(f"  (patience {patience_cnt}/{args.patience})")

        if patience_cnt >= args.patience:
            print(f"\n⏹  Early Stopping! best val_acc={best_val_acc:.4f}")
            break

    print(f"\n{'='*55}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    te_loss, te_acc = evaluate(model, loaders['test'], criterion, device)
    print(f"테스트 accuracy: {te_acc:.4f}")
    print(f"{'='*55}")

    return history


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--faces_dir',   default='/content/data/faces/real_vs_fake/real-vs-fake')
    parser.add_argument('--pixiv_dir',   default='/content/data/pixiv/aidataset')
    parser.add_argument('--intel_dir',   default='/content/data/intel')
    parser.add_argument('--ai_gen_dir',  default='/content/data/ai_generated/labeled_images')
    parser.add_argument('--coco_dir',    default='/content/data/coco')
    parser.add_argument('--mj_dir',      default='/content/data/midjourney/images')
    parser.add_argument('--csv_path',    default='/content/data/aivshuman/train_data/train.csv')
    parser.add_argument('--csv_img_dir', default='/content/data/aivshuman/train_data')
    parser.add_argument('--weights_dir', default='weights')
    parser.add_argument('--use_fft',     action='store_true')
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=32)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--patience',    type=int,   default=5)
    parser.add_argument('--coco_sample', type=int,   default=10000)
    args = parser.parse_args()
    train(args)
