"""
app.py
------
Flask 웹 서버.
이미지 업로드 → 모델 추론 → 결과(JSON) 반환.

실행:
    python web/app.py

접속:
    http://localhost:5000

Colab에서 외부 접속이 필요한 경우 ngrok 사용:
    !pip install pyngrok
    from pyngrok import ngrok
    ngrok.set_auth_token("YOUR_TOKEN")
    public_url = ngrok.connect(5000)
    print(public_url)
"""

import io
import os
import sys
import base64

import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image

# src 폴더 경로 추가
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))

from model import FakeDetector
from gradcam import run_gradcam

app = Flask(__name__,
            template_folder='templates',
            static_folder='static')

# ── 전역 모델 (서버 시작 시 1회 로딩) ──────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = None


def load_model():
    global model

    # 체크포인트 경로 우선순위:
    #   1. Google Drive (Colab)
    #   2. 로컬 weights/ 폴더
    candidates = [
        '/content/drive/MyDrive/fakescope_checkpoints/best_model.pth',
        os.path.join(ROOT, 'weights', 'best_model.pth'),
    ]

    ckpt_path = None
    for p in candidates:
        if os.path.exists(p):
            ckpt_path = p
            break

    if ckpt_path is None:
        print("⚠️  모델 가중치 없음. 랜덤 가중치로 구조만 로드합니다.")
        print("   학습 먼저 실행: python src/train.py --use_fft")
        model = FakeDetector(use_freq=True).to(device)
        model.eval()
        return

    ckpt     = torch.load(ckpt_path, map_location=device)
    use_freq = ckpt.get('use_freq', True)
    model    = FakeDetector(use_freq=use_freq).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    val_acc = ckpt.get('val_acc', 0)
    print(f"✅ 모델 로드 완료 | 경로: {ckpt_path} | val_acc={val_acc:.4f}")


# ── 라우트 ──────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    POST /predict
    Content-Type: multipart/form-data
    Field: 'image' (이미지 파일)

    응답 JSON:
    {
        "label"      : "Real" | "Fake",
        "confidence" : 0.97,
        "real_prob"  : 0.03,
        "fake_prob"  : 0.97,
        "gradcam_b64": "<base64 JPEG>"
    }
    """
    if 'image' not in request.files:
        return jsonify({'error': '이미지가 없습니다.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '파일명이 비어있습니다.'}), 400

    try:
        pil_img = Image.open(file.stream).convert('RGB')

        # 추론
        overlay_np, pred_label, confidence, real_prob, fake_prob = \
            run_gradcam(model, pil_img, device)

        # GradCAM 이미지를 base64 인코딩해서 JSON에 포함
        buf = io.BytesIO()
        Image.fromarray(overlay_np).save(buf, format='JPEG', quality=85)
        gradcam_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        return jsonify({
            'label'      : pred_label,
            'confidence' : round(confidence, 4),
            'real_prob'  : round(real_prob, 4),
            'fake_prob'  : round(fake_prob, 4),
            'gradcam_b64': gradcam_b64,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ── 실행 ────────────────────────────────────────────────────
if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=False)
