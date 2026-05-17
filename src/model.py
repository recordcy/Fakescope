"""
model.py
--------
MobileNetV3-Large (RGB Branch) +
FFT 기반 주파수 분석 CNN (Frequency Branch) 를
합친 최종 분류 모델.

EfficientNet-B4 대신 MobileNetV3를 쓰는 이유:
    - 파라미터 수: EfficientNet-B4 약 19M → MobileNetV3 약 5.4M
    - 추론 속도 약 3배 빠름
    - Colab 무료 메모리(12GB) 안에서 batch_size=32 안정적으로 돌아감
    - 정확도 차이: 약 2~3% (텀프로젝트 수준에서 충분)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ────────────────────────────────────────────────────────────
# 1. Frequency Branch
# ────────────────────────────────────────────────────────────
class FrequencyBranch(nn.Module):
    """
    AI 생성 이미지는 픽셀 공간에서는 사실적으로 보여도
    주파수 도메인(FFT)에서 특유의 격자 패턴(grid artifact)이 나타납니다.
    이 branch가 그 패턴을 학습합니다.

    흐름:
        RGB 이미지
        → 그레이스케일
        → 2D FFT
        → log magnitude (저주파 중앙 정렬)
        → 소형 CNN
        → 128-dim 특징 벡터
    """

    def __init__(self, out_dim: int = 128):
        super().__init__()

        self.cnn = nn.Sequential(
            # 입력: (B, 1, 224, 224)
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # → 112

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                 # → 56

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),         # → (B, 64, 4, 4)
        )

        self.fc = nn.Linear(64 * 4 * 4, out_dim)

    def forward(self, x):
        # x: (B, 3, 224, 224) — 정규화된 RGB 텐서

        # 그레이스케일 (채널 평균)
        gray = x.mean(dim=1, keepdim=True)              # (B, 1, 224, 224)

        # FFT → magnitude spectrum
        fft       = torch.fft.fft2(gray)
        magnitude = torch.abs(fft)
        magnitude = torch.fft.fftshift(magnitude)       # 저주파를 중앙으로
        magnitude = torch.log(magnitude + 1e-8)         # 로그 스케일

        # 배치 내 각 샘플을 [0, 1] 로 정규화
        B = magnitude.shape[0]
        mn = magnitude.view(B, -1).min(1)[0].view(B, 1, 1, 1)
        mx = magnitude.view(B, -1).max(1)[0].view(B, 1, 1, 1)
        magnitude = (magnitude - mn) / (mx - mn + 1e-8)

        feat = self.cnn(magnitude)                      # (B, 64, 4, 4)
        feat = feat.view(B, -1)                         # (B, 1024)
        feat = self.fc(feat)                            # (B, 128)
        return feat


# ────────────────────────────────────────────────────────────
# 2. RGB Branch
# ────────────────────────────────────────────────────────────
class RGBBranch(nn.Module):
    """
    MobileNetV3-Large (ImageNet 사전학습) 백본.
    마지막 분류기를 제거하고 특징 벡터(960-dim)만 추출합니다.
    """

    def __init__(self):
        super().__init__()
        base = models.mobilenet_v3_large(
            weights=models.MobileNet_V3_Large_Weights.DEFAULT)

        # features + avgpool 만 사용 (classifier 제거)
        self.features  = base.features
        self.pool      = base.avgpool
        self.out_dim   = 960

    def forward(self, x):
        x = self.features(x)    # (B, 960, 7, 7)
        x = self.pool(x)        # (B, 960, 1, 1)
        x = x.flatten(1)        # (B, 960)
        return x


# ────────────────────────────────────────────────────────────
# 3. 최종 분류 모델
# ────────────────────────────────────────────────────────────
class FakeDetector(nn.Module):
    """
    RGB Branch + Frequency Branch 특징을 이어붙여 분류.

    use_freq=False 로 설정하면 RGB Branch만 사용 → ablation 실험용
    """

    def __init__(self, use_freq: bool = True):
        super().__init__()
        self.use_freq = use_freq

        self.rgb_branch  = RGBBranch()
        self.freq_branch = FrequencyBranch(out_dim=128) if use_freq else None

        in_dim = self.rgb_branch.out_dim + (128 if use_freq else 0)
        # 960 + 128 = 1088  (use_freq=True 기준)

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),   # 0=Real, 1=Fake
        )

    def forward(self, x):
        rgb_feat = self.rgb_branch(x)

        if self.use_freq:
            freq_feat = self.freq_branch(x)
            feat = torch.cat([rgb_feat, freq_feat], dim=1)  # (B, 1088)
        else:
            feat = rgb_feat                                  # (B, 960)

        return self.classifier(feat)                        # (B, 2)


# ────────────────────────────────────────────────────────────
# 동작 확인
# ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    model = FakeDetector(use_freq=True)
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    print(f"출력 shape: {out.shape}")   # (2, 2)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"전체 파라미터:    {total:,}")
    print(f"학습 가능 파라미터: {trainable:,}")
