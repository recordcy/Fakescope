"""
dataset.py
----------
4차 시도 — Fake 데이터 다양성 강화

Real(0):
  ① 140k 얼굴 real
  ② Pixiv illust
  ③ Intel Image Classification
  ④ COCO 10,000장

Fake(1):
  ① 140k 얼굴 fake
  ② Pixiv ai
  ③ AI Generated Images (11,300장)
  ④ Midjourney Images (신규) ← 실사 스타일 AI 이미지
  ⑤ AI vs Human (CSV, 신규) ← 다양한 실사 AI 이미지

핵심 변경:
  - Fake에 실사 스타일 AI 이미지 대폭 추가
  - Midjourney: 숫자 폴더 구조 → 재귀 수집
  - alessandrasala79: CSV 파싱으로 fake만 추출
"""

import csv
import io
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ────────────────────────────────────────────────────────────
# 1. 커스텀 Transform
# ────────────────────────────────────────────────────────────
class RandomJPEGCompression:
    def __init__(self, quality_range=(40, 95), p=0.5):
        self.quality_range = quality_range
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        quality = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()


class RandomGaussianNoise:
    def __init__(self, std_range=(0.01, 0.05), p=0.3):
        self.std_range = std_range
        self.p = p

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor
        std = random.uniform(*self.std_range)
        noise = torch.randn_like(tensor) * std
        return torch.clamp(tensor + noise, 0, 1)


# ────────────────────────────────────────────────────────────
# 2. 전처리 변환
# ────────────────────────────────────────────────────────────
def get_transforms(phase: str):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if phase == 'train':
        return transforms.Compose([
            RandomJPEGCompression(quality_range=(40, 95), p=0.5),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomRotation(20),
            transforms.ColorJitter(
                brightness=0.3, contrast=0.3,
                saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))],
                p=0.3),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            RandomGaussianNoise(std_range=(0.01, 0.05), p=0.3),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ────────────────────────────────────────────────────────────
# 3. 이미지 수집 함수들
# ────────────────────────────────────────────────────────────
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}


def collect_images(folder: str, label: int,
                   recursive: bool = False,
                   max_count: int = None):
    p = Path(folder)
    if not p.exists():
        print(f"[경고] 폴더 없음: {folder} → 건너뜀")
        return []
    files = list(p.rglob('*') if recursive else p.glob('*'))
    files = [f for f in files if f.suffix.lower() in EXTENSIONS]
    if max_count and len(files) > max_count:
        random.shuffle(files)
        files = files[:max_count]
    return [(str(f), label) for f in files]


def collect_intel(intel_dir: str, phase: str, label: int = 0):
    base = Path(intel_dir) / (
        'seg_train/seg_train' if phase == 'train' else 'seg_test/seg_test')
    if not base.exists():
        print(f"[경고] Intel 폴더 없음: {base} → 건너뜀")
        return []
    files = [f for f in base.rglob('*') if f.suffix.lower() in EXTENSIONS]
    return [(str(f), label) for f in files]


def collect_coco(coco_dir: str, phase: str,
                 label: int = 0, max_count: int = 10000):
    sub = 'train2017' if phase == 'train' else 'val2017'
    base = Path(coco_dir) / 'coco_data' / 'images' / sub
    if not base.exists():
        base = Path(coco_dir) / 'images' / sub
    if not base.exists():
        print(f"[경고] COCO 폴더 없음: {base} → 건너뜀")
        return []
    files = [f for f in base.glob('*') if f.suffix.lower() in EXTENSIONS]
    random.shuffle(files)
    n = max_count if phase == 'train' else max(1000, max_count // 10)
    files = files[:n]
    print(f"  COCO ({phase}): {len(files):,}장")
    return [(str(f), label) for f in files]


def collect_midjourney(mj_dir: str, phase: str,
                       label: int = 1, max_count: int = None):
    """
    Midjourney 데이터셋: images/0/, images/1/, ... 숫자 폴더 구조
    전체 재귀 수집 후 phase별로 분할
    """
    base = Path(mj_dir)
    if not base.exists():
        print(f"[경고] Midjourney 폴더 없음: {base} → 건너뜀")
        return []

    all_files = [f for f in base.rglob('*')
                 if f.suffix.lower() in EXTENSIONS]
    random.Random(42).shuffle(all_files)  # 시드 고정으로 항상 같은 분할

    # train 70% / valid 15% / test 15%
    n = len(all_files)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)

    if phase == 'train':
        files = all_files[:n_train]
    elif phase == 'valid':
        files = all_files[n_train:n_train + n_valid]
    else:
        files = all_files[n_train + n_valid:]

    if max_count and len(files) > max_count:
        files = files[:max_count]

    print(f"  Midjourney ({phase}): {len(files):,}장")
    return [(str(f), label) for f in files]


def collect_csv_dataset(csv_path: str, img_dir: str,
                        label_col: str, fake_label,
                        img_col: str, phase: str,
                        label: int = 1, max_count: int = None):
    """
    alessandrasala79/ai-vs-human-generated-dataset:
    CSV 파일에서 fake 이미지만 추출

    csv_path  : train.csv 또는 test.csv 경로
    img_dir   : 이미지 루트 폴더
    label_col : 레이블 컬럼명 (예: 'label')
    fake_label: fake를 나타내는 값 (예: 0 또는 'fake')
    img_col   : 이미지 경로 컬럼명 (예: 'file_name')
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"[경고] CSV 없음: {csv_path} → 건너뜀")
        return []

    samples = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row[label_col]) == str(fake_label):
                img_path = Path(img_dir) / row[img_col]
                if img_path.exists():
                    samples.append((str(img_path), label))

    random.Random(42).shuffle(samples)

    # train/valid/test 분할
    n = len(samples)
    n_train = int(n * 0.70)
    n_valid = int(n * 0.15)

    if phase == 'train':
        samples = samples[:n_train]
    elif phase == 'valid':
        samples = samples[n_train:n_train + n_valid]
    else:
        samples = samples[n_train + n_valid:]

    if max_count and len(samples) > max_count:
        samples = samples[:max_count]

    print(f"  CSV Fake ({phase}): {len(samples):,}장")
    return samples


# ────────────────────────────────────────────────────────────
# 4. 통합 Dataset 클래스
# ────────────────────────────────────────────────────────────
class FakeDetectDataset(Dataset):
    PIXIV_PHASE = {'train': 'train', 'valid': 'val', 'test': 'test'}

    def __init__(self, faces_dir, pixiv_dir, intel_dir,
                 ai_gen_dir, coco_dir, mj_dir,
                 csv_path, csv_img_dir,
                 phase, coco_sample=10000):
        self.transform = get_transforms(phase)
        pphase = self.PIXIV_PHASE[phase]

        # ── Real ──
        faces_real = collect_images(f"{faces_dir}/{phase}/real", 0)
        pixiv_real = collect_images(f"{pixiv_dir}/{pphase}/illust", 0)
        intel_real = collect_intel(intel_dir, phase, 0)
        coco_real  = collect_coco(coco_dir, phase, 0, coco_sample)

        # ── Fake ──
        faces_fake = collect_images(f"{faces_dir}/{phase}/fake", 1)
        pixiv_fake = collect_images(f"{pixiv_dir}/{pphase}/ai", 1)

        n_ai = None if phase == 'train' else 300
        ai_fake  = collect_images(ai_gen_dir, 1, recursive=True,
                                   max_count=n_ai)
        mj_fake  = collect_midjourney(mj_dir, phase, 1)
        csv_fake = collect_csv_dataset(
            csv_path, csv_img_dir,
            label_col='label', fake_label=0,
            img_col='file_name', phase=phase)

        all_real = faces_real + pixiv_real + intel_real + coco_real
        all_fake = (faces_fake + pixiv_fake + ai_fake
                    + mj_fake + csv_fake)

        # 1:1 균형 (얼굴에서만 자르기)
        if len(all_real) > len(all_fake):
            shortage = len(all_real) - len(all_fake)
            random.shuffle(faces_real)
            faces_real = faces_real[shortage:]
            all_real = faces_real + pixiv_real + intel_real + coco_real

        min_count = min(len(all_real), len(all_fake))
        random.shuffle(all_real)
        random.shuffle(all_fake)
        all_real = all_real[:min_count]
        all_fake = all_fake[:min_count]

        self.samples = all_real + all_fake
        random.shuffle(self.samples)

        print(f"\n[{phase}] 최종 구성:")
        print(f"  Real → 얼굴:{len(faces_real):,} | 일러스트:{len(pixiv_real):,} | "
              f"Intel:{len(intel_real):,} | COCO:{len(coco_real):,}")
        print(f"  Fake → 얼굴:{len(faces_fake):,} | 일러스트:{len(pixiv_fake):,} | "
              f"AI생성:{len(ai_fake):,} | MJ:{len(mj_fake):,} | CSV:{len(csv_fake):,}")
        print(f"  균형 후 → Real={min_count:,} Fake={min_count:,} "
              f"Total={len(self.samples):,}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            img = self.transform(img)
        except Exception:
            return self.__getitem__((idx + 1) % len(self.samples))
        return img, torch.tensor(label, dtype=torch.long)


# ────────────────────────────────────────────────────────────
# 5. DataLoader
# ────────────────────────────────────────────────────────────
def get_dataloaders(faces_dir, pixiv_dir, intel_dir,
                    ai_gen_dir, coco_dir, mj_dir,
                    csv_path, csv_img_dir,
                    batch_size=32, num_workers=2,
                    coco_sample=10000):
    loaders = {}
    for phase in ('train', 'valid', 'test'):
        ds = FakeDetectDataset(
            faces_dir, pixiv_dir, intel_dir,
            ai_gen_dir, coco_dir, mj_dir,
            csv_path, csv_img_dir,
            phase, coco_sample)
        loaders[phase] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(phase == 'train'),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    return loaders
