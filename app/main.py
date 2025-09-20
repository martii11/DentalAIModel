# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional
from pathlib import Path
import os, uuid, mimetypes, asyncio, shutil

import requests
import httpx
from PIL import Image
import numpy as np
import cv2

from ultralytics import YOLO

# ====== HF: tek seferlik indir & yerel dosyadan yükle (senin eski projeye benzer) ======
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# HF URL (public) -> doğrudan raw dosya linki (senin beyin_tumoru_modeli.h5 örneği gibi)
HF_MODEL_URL = os.getenv("HF_MODEL_URL", "https://huggingface.co/Mortines/dentalDetect/resolve/main/best.pt")
LOCAL_MODEL_PATH = MODELS_DIR / "best.pt"

def ensure_model_present():
    if LOCAL_MODEL_PATH.exists() and LOCAL_MODEL_PATH.stat().st_size > 0:
        return
    print("📥 Hugging Face'ten YOLO model indiriliyor...")
    r = requests.get(HF_MODEL_URL, stream=True, timeout=60)
    r.raise_for_status()
    with open(LOCAL_MODEL_PATH, "wb") as f:
        shutil.copyfileobj(r.raw, f)
    print("✅ Model indirildi:", LOCAL_MODEL_PATH)

# ====== Klasörler ======
DATA_ROOT = Path(os.getenv("DATA_ROOT", BASE_DIR / "images"))
RAW_DIR = DATA_ROOT / "raw"
OUT_DIR = DATA_ROOT / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ====== Dış upload endpoint (IFormFile) ======
UPLOAD_ENDPOINT = os.getenv("UPLOAD_ENDPOINT", "https://apiimage.kuryemio.com.tr/api/Image/AddImage")

# ====== Eşikler ======
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))
IMGSZ = int(os.getenv("IMGSZ", "640"))

# ====== Sınıf adları TR map ve renkler ======
TR_MAP = {
    "BDC-BDR": "BDC-BDR",
    "Caries": "Curuk",
    "Fractured Teeth": "Kirik",
    "Healthy Teeth": "Saglikli",
    "Impacted teeth": "Gomulu",
    "Infection": "Enfeksiyon",
}
COLOR_MAP = {
    "BDC-BDR":    (128,   0,   0),
    "Curuk":      (  0, 128, 128),
    "Kirik":      (  0, 215, 255),
    "Saglikli":   (255, 255,   0),
    "Gomulu":     (211,  85, 186),
    "Enfeksiyon": (128, 128,   0),
}

def draw_turkish_annotations(orig_rgb: np.ndarray, res, names):
    if orig_rgb.ndim == 2:
        orig_rgb = cv2.cvtColor(orig_rgb, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    img = np.ascontiguousarray(img, dtype=np.uint8)
    H, W = img.shape[:2]

    boxes = getattr(res, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    xyxy = boxes.xyxy.cpu().numpy()
    clss = boxes.cls.cpu().numpy().astype(int)
    confs = boxes.conf.cpu().numpy()

    base_scale = max(0.5, min(1.2, W / 1024 * 0.8))
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick_box  = max(2, int(2 * base_scale)) + 1
    thick_text = max(1, int(base_scale * 1.2))
    pad = max(4, int(4 * base_scale))

    order = np.argsort(xyxy[:, 1])
    occupied = []

    def overlap(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        return not (ax2 < bx1 or bx2 < ax1 or ay2 < by1 or by2 < ay1)

    for i in order:
        x1, y1, x2, y2 = map(int, xyxy[i])
        cid = clss[i]; conf = confs[i]

        name_en = names[cid]
        name_tr = TR_MAP.get(name_en, name_en)
        label = f"{name_tr} {conf:.2f}"
        color = COLOR_MAP.get(name_tr, (0, 140, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick_box)
        (tw, th), _ = cv2.getTextSize(label, font, base_scale, thick_text)

        lx1 = x1
        ly2 = y1 - 2
        ly1 = ly2 - th - 2 * pad
        if ly1 < 0:
            ly1 = y1 + 2
            ly2 = ly1 + th + 2 * pad
        lx2 = min(W - 1, lx1 + tw + 2 * pad)

        tries = 0
        box_lab = [lx1, ly1, lx2, ly2]
        while any(overlap(box_lab, occ) for occ in occupied) and tries < 20:
            shift = th + 3 * pad
            box_lab[1] = min(H - th - 2*pad, box_lab[1] + shift)
            box_lab[3] = box_lab[1] + th + 2*pad
            tries += 1

        lx1, ly1, lx2, ly2 = map(int, box_lab)
        occupied.append([lx1, ly1, lx2, ly2])

        cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, -1)
        brightness = 0.299*color[2] + 0.587*color[1] + 0.114*color[0]
        text_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)
        cv2.putText(img, label, (lx1 + pad, ly2 - pad),
                    font, base_scale, text_color, thick_text, cv2.LINE_AA)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def guess_ext_from_content_type(ct: str, fallback=".jpg"):
    if not ct:
        return fallback
    ext = mimetypes.guess_extension(ct.split(";")[0].strip())
    if ext in (None, ".jpe"):
        return fallback
    if ext == ".jpeg":
        return ".jpg"
    return ext

# ====== FastAPI ======
app = FastAPI(title="BrainAI Inference API", version="1.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model: Optional[YOLO] = None

class PredictIn(BaseModel):
    url: HttpUrl

class XRayPredictionResponse(BaseModel):
    prediction: str
    confidence: str
    ImageUrl: str

@app.on_event("startup")
def startup():
    # 1) Model dosyasını HF'den indir (yoksa)
    ensure_model_present()
    # 2) Yükle
    global _model
    _model = YOLO(str(LOCAL_MODEL_PATH))

@app.get("/health")
def health():
    return {"status": "ok", "model_path": str(LOCAL_MODEL_PATH)}

async def download_image_to_disk(url: str) -> Path:
    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        r = await client.get(url)
        if r.status_code != 200:
            raise HTTPException(400, f"Görsel indirilemedi: HTTP {r.status_code}")
        content_type = r.headers.get("Content-Type", "").lower()
        ext = guess_ext_from_content_type(content_type)
        fname = f"{uuid.uuid4().hex}{ext}"
        out_path = RAW_DIR / fname
        out_path.write_bytes(r.content)
        return out_path

def infer_and_annotate(img_path: Path) -> tuple[Path, str, float]:
    assert _model is not None, "Model yüklenemedi"
    pil = Image.open(img_path).convert("RGB")
    arr = np.array(pil)

    results = _model.predict(arr, conf=CONF_THRESHOLD, imgsz=IMGSZ, verbose=False)
    res = results[0]

    top_label_tr = "No Detection"
    top_conf = 0.0
    if getattr(res, "boxes", None) is not None and len(res.boxes):
        confs = res.boxes.conf.cpu().numpy()
        clss = res.boxes.cls.cpu().numpy().astype(int)
        i = int(np.argmax(confs))
        top_conf = float(confs[i])
        name_en = _model.names[int(clss[i])]
        top_label_tr = TR_MAP.get(name_en, name_en)

    annot = draw_turkish_annotations(arr, res, _model.names)
    out_path = OUT_DIR / f"{img_path.stem}_annot.jpg"
    Image.fromarray(annot).save(out_path, "JPEG", quality=95)
    return out_path, top_label_tr, top_conf

async def upload_processed_image(file_path: Path) -> str:
    timeout = httpx.Timeout(20.0, connect=5.0)
    mime = mimetypes.guess_type(str(file_path))[0] or "image/jpeg"
    files = {"file": (file_path.name, file_path.open("rb"), mime)}
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(UPLOAD_ENDPOINT, files=files)
        if resp.status_code != 200:
            raise HTTPException(502, f"Yükleme başarısız: HTTP {resp.status_code} - {resp.text}")
        data = resp.json()
        image_url = data.get("ImageUrl") or data.get("imageUrl")
        if not image_url:
            raise HTTPException(502, "Yükleme yanıtında ImageUrl bulunamadı")
        return image_url

@app.post("/predict", response_model=XRayPredictionResponse)
async def predict(payload: PredictIn):
    try:
        img_path = await download_image_to_disk(str(payload.url))
    except Exception as e:
        raise HTTPException(400, f"Görsel indirilemedi: {e}")

    try:
        annot_path, label_tr, conf = await asyncio.to_thread(infer_and_annotate, img_path)
    except Exception as e:
        raise HTTPException(500, f"Model tahmini hatası: {e}")

    try:
        image_url = await upload_processed_image(annot_path)
    except Exception as e:
        raise HTTPException(502, f"Görsel yükleme hatası: {e}")

    return XRayPredictionResponse(
        prediction=label_tr,
        confidence=f"{conf:.2f}",
        ImageUrl=image_url,
    )
