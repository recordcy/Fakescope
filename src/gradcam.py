"""
gradcam.py
----------
GradCAM: 모델이 이미지의 어느 영역을 보고 판단했는지 히트맵으로 표시.

발표 때 "왜 Fake냐?" 를 시각적으로 설명하는 핵심 도구입니다.

동작 원리:
    1. 지정 레이어의 feature map을 forward hook으로 저장
    2. 해당 클래스에 대한 gradient를 backward hook으로 저장
    3. gradient를 Global Average Pooling → 채널별 가중치
    4. feature map × 가중치 합산 → 히트맵
    5. ReLU → 음수 제거 (양의 기여만 표시)
    6. 원본 이미지 크기로 bilinear upsampling
"""

import io
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


# ────────────────────────────────────────────────────────────
# GradCAM 클래스
# ────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model: torch.nn.Module,
                 target_layer: torch.nn.Module):
        """
        model        : 학습된 FakeDetector
        target_layer : GradCAM을 적용할 레이어
                       (보통 마지막 Conv 레이어)
        """
        self.model       = model
        self.model.eval()
        self._feature    = None
        self._gradient   = None

        # forward hook: feature map 캡처
        self._fwd = target_layer.register_forward_hook(
            lambda m, inp, out: setattr(self, '_feature', out.detach())
        )
        # backward hook: gradient 캡처
        self._bwd = target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, '_gradient', go[0].detach())
        )

    def generate(self, img_tensor: torch.Tensor, class_idx: int = None):
        """
        img_tensor : (1, 3, 224, 224) 정규화된 텐서
        class_idx  : None이면 예측 클래스 자동 선택

        반환:
            cam        : (224, 224) float32, [0, 1]
            pred_label : 'Real' | 'Fake'
            confidence : float, 해당 클래스 확률
        """
        device = next(self.model.parameters()).device
        img_tensor = img_tensor.to(device)

        # Forward
        logits = self.model(img_tensor)
        probs  = torch.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = logits.argmax(1).item()

        # Backward
        self.model.zero_grad()
        logits[0, class_idx].backward()

        # CAM 계산
        grad = self._gradient[0]                        # (C, H, W)
        feat = self._feature[0]                         # (C, H, W)
        weights = grad.mean(dim=(1, 2))                 # (C,)
        cam = (weights[:, None, None] * feat).sum(0)    # (H, W)
        cam = F.relu(cam)

        # 224×224 upsampling
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze().cpu().numpy()

        # 정규화
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        pred_label  = 'Fake' if class_idx == 1 else 'Real'
        confidence  = probs[0, class_idx].item()
        return cam, pred_label, confidence

    def overlay(self, original: np.ndarray, cam: np.ndarray,
                alpha: float = 0.45) -> np.ndarray:
        """
        원본 이미지에 GradCAM 히트맵을 반투명 오버레이.

        original : (224, 224, 3) uint8
        cam      : (224, 224) float [0, 1]
        alpha    : 히트맵 투명도
        """
        heatmap = cv2.applyColorMap(
            (cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        result  = (1 - alpha) * original + alpha * heatmap
        return np.clip(result, 0, 255).astype(np.uint8)

    def remove_hooks(self):
        """메모리 누수 방지 — 사용 후 반드시 호출."""
        self._fwd.remove()
        self._bwd.remove()


# ────────────────────────────────────────────────────────────
# 편의 함수 — app.py 에서 호출
# ────────────────────────────────────────────────────────────
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


def run_gradcam(model, image_input, device: torch.device):
    """
    image_input : 파일 경로(str) 또는 PIL Image

    반환:
        overlay_np  : (224, 224, 3) uint8 numpy — GradCAM 오버레이 이미지
        pred_label  : 'Real' | 'Fake'
        confidence  : float
        real_prob   : float
        fake_prob   : float
    """
    # PIL 이미지 로딩
    if isinstance(image_input, str):
        pil = Image.open(image_input).convert('RGB')
    else:
        pil = image_input.convert('RGB')

    pil_224    = pil.resize((224, 224))
    original   = np.array(pil_224)
    img_tensor = _transform(pil).unsqueeze(0)

    # GradCAM 대상 레이어: MobileNetV3 마지막 Conv block
    target_layer = model.rgb_branch.features[-1]

    gc = GradCAM(model, target_layer)
    cam, pred_label, confidence = gc.generate(img_tensor)
    overlay = gc.overlay(original, cam)
    gc.remove_hooks()

    # 전체 확률값
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        probs  = torch.softmax(logits, dim=1)[0].cpu().tolist()

    return overlay, pred_label, confidence, probs[0], probs[1]
