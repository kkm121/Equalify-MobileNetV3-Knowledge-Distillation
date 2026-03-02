import torch
import cv2
import numpy as np
import os
import sys

CONFIG = {
    'IMG_SIZE': 224,
    'DEVICE': 'cpu',
    'MODEL_PATH': os.path.join(os.path.dirname(__file__), 'model', 'model.pth'),
    'TTA_LEVEL': 1
}

torch.set_num_threads(4)

MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        try:
            if not os.path.exists(CONFIG['MODEL_PATH']):
                return None
            MODEL = torch.jit.load(CONFIG['MODEL_PATH'], map_location='cpu')
            MODEL.eval()
        except Exception:
            return None
    return MODEL

def process_view(img):
    img = cv2.resize(img, (CONFIG['IMG_SIZE'], CONFIG['IMG_SIZE']), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = img.transpose(2, 0, 1)
    return torch.from_numpy(img).unsqueeze(0).float()

def predict(image_path):
    model = get_model()
    if model is None:
        return 0, 0.5

    try:
        raw_img = cv2.imread(image_path)
        if raw_img is None: return 0, 0.5
        raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    except:
        return 0, 0.5

    tensors = []
    tensors.append(process_view(raw_img))

    if CONFIG['TTA_LEVEL'] >= 2:
        tensors.append(process_view(cv2.flip(raw_img, 1)))

    if CONFIG['TTA_LEVEL'] == 4:
        h, w = raw_img.shape[:2]
        ch, cw = int(h * 0.9), int(w * 0.9)
        dy, dx = (h - ch) // 2, (w - cw) // 2
        zoom = raw_img[dy:dy+ch, dx:dx+cw]
        tensors.append(process_view(zoom))
        tensors.append(process_view(cv2.flip(zoom, 1)))

    probs = []
    with torch.no_grad():
        for t in tensors:
            logits = model(t)
            probs.append(torch.sigmoid(logits).item())

    prob = sum(probs) / len(probs)

    if prob > 0.5:
        return 1, float(prob)
    else:
        return 0, float(1.0 - prob)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        lbl, conf = predict(sys.argv[1])
        print(f"Label: {lbl}, Confidence: {conf:.4f}")
