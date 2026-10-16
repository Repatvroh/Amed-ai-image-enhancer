import os
FILES = {
"backend/config.py": '''import os
from dotenv import load_dotenv
load_dotenv()
API_SECRET_KEY = os.getenv("API_SECRET_KEY", "")
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
MAX_INPUT_PIXELS = int(os.getenv("MAX_INPUT_PIXELS", "4000000"))
FREE_TIER_LIMIT = int(os.getenv("FREE_TIER_LIMIT", "5"))
FILE_TTL_SECONDS = int(os.getenv("FILE_TTL_SECONDS", "3600"))
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "http://localhost:8000")
''',
"backend/requirements.txt": '''fastapi==0.115.0
uvicorn[standard]==0.32.0
python-multipart==0.0.12
pillow==11.0.0
numpy==2.1.0
torch==2.5.0
torchvision==0.20.0
realesrgan==0.3.0
gfpgan==1.3.8
basicsr==1.4.2
opencv-python-headless==4.10.0
python-dotenv==1.0.1
''',
"backend/upscaler.py": '''import numpy as np
from PIL import Image
import io
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

class Upscaler:
    def __init__(self):
        self.models = {}
        self._load()
    def _load(self):
        m4 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.models["x4"] = RealESRGANer(scale=4, model_path="weights/RealESRGAN_x4plus.pth", model=m4, tile=400, tile_pad=10, pre_pad=0, half=False)
        m2 = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        self.models["x2"] = RealESRGANer(scale=2, model_path="weights/RealESRGAN_x2plus.pth", model=m2, tile=400, tile_pad=10, pre_pad=0, half=False)
        mg = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        self.models["general"] = RealESRGANer(scale=4, model_path="weights/realesr-general-x4v3.pth", model=mg, tile=400, tile_pad=10, pre_pad=0, half=False)
    def analyze_image(self, img):
        w, h = img.size
        px = w * h
        return {"width": w, "height": h, "pixels": px, "megapixels": round(px/1_000_000, 2), "max_safe_scale": 4 if px <= 2_000_000 else 2}
    def upscale(self, image_bytes, target="4x", face_enhance=True):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        key = "x2" if target == "2x" else "general"
        out, _ = self.models[key].enhance(np.array(img), outscale=4 if key != "x2" else 2, face_enhance=face_enhance)
        result = Image.fromarray(out)
        if target == "auto_2k": result = self._resize(result, 2560)
        elif target == "auto_4k": result = self._resize(result, 3840)
        elif target == "auto_8k": result = self._resize(result, 7680)
        buf = io.BytesIO()
        result.save(buf, format="PNG", optimize=True)
        return buf.getvalue()
    def _resize(self, img, tw):
        w, h = img.size
        if w >= tw: return img
        nh = int(h * tw / w)
        if tw * nh > 67_108_864: return img
        return img.resize((tw, nh), Image.LANCZOS)
''',
"backend/face_restore.py": '''import io
import numpy as np
from PIL import Image
from gfpgan import GFPGANer

class FaceRestorer:
    def __init__(self):
        self.restorer = GFPGANer(model_path="weights/GFPGANv1.4.pth", upscale=2, arch="clean", channel_multiplier=2, bg_upsampler=None)
    def restore(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        _, _, out = self.restorer.enhance(np.array(img), has_aligned=False, only_center_face=False, paste_back=True)
        buf = io.BytesIO()
        Image.fromarray(out).save(buf, format="PNG")
        return buf.getvalue()
''',
"backend/main.py": '''import uuid, time, asyncio
from io import BytesIO
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from config import *
from upscaler import Upscaler
from face_restore import FaceRestorer

app = FastAPI(title="AMED AI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
upscaler = None
face_restorer = None
files_store = {}
user_usage = {}

@app.on_event("startup")
async def startup():
    global upscaler, face_restorer
    upscaler = Upscaler()
    try: face_restorer = FaceRestorer()
    except Exception as e: print(f"GFPGAN skip: {e}")
    asyncio.create_task(cleanup_loop())

async def cleanup_loop():
    while True:
        await asyncio.sleep(60)
        now = time.time()
        for k in [k for k,v in files_store.items() if now - v["created"] > FILE_TTL_SECONDS]:
            del files_store[k]

async def verify_token(authorization: str = Header(None)):
    if not authorization: raise HTTPException(401, "Token manquant")
    if authorization.replace("Bearer ", "") != API_SECRET_KEY: raise HTTPException(403, "Token invalide")

def get_client_id(authorization: str = Header(None)) -> str:
    return (authorization or "anon").replace("Bearer ", "")[:16]

@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": upscaler is not None}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), _: str = Depends(verify_token)):
    content = await file.read()
    return upscaler.analyze_image(Image.open(BytesIO(content)))

@app.post("/upscale")
async def upscale(file: UploadFile = File(...), target: str = Form("4x"), face_enhance: bool = Form(True), _: str = Depends(verify_token), client_id: str = Depends(get_client_id)):
    usage = user_usage.get(client_id, 0)
    if usage >= FREE_TIER_LIMIT: raise HTTPException(402, "Quota atteint")
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_MB * 1024 * 1024: raise HTTPException(413, "Trop gros")
    try:
        loop = asyncio.get_event_loop()
        out = await loop.run_in_executor(None, upscaler.upscale, content, target, face_enhance)
        if face_enhance and face_restorer:
            try: out = await loop.run_in_executor(None, face_restorer.restore, out)
            except: pass
        fid = str(uuid.uuid4())
        files_store[fid] = {"data": out, "created": time.time(), "content_type": "image/png"}
        user_usage[client_id] = usage + 1
        return {"file_id": fid, "download_url": f"{PUBLIC_BASE_URL}/download/{fid}", "remaining_free": max(0, FREE_TIER_LIMIT - user_usage[client_id])}
    except Exception as e: raise HTTPException(500, str(e))

@app.get("/download/{fid}")
async def download(fid: str):
    if fid not in files_store: raise HTTPException(404, "Introuvable")
    e = files_store[fid]
    return FileResponse(BytesIO(e["data"]), media_type=e["content_type"], filename=f"amed_{fid[:8]}.png")

@app.delete("/files/{fid}")
async def delete_file(fid: str, _: str = Depends(verify_token)):
    files_store.pop(fid, None)
    return {"status": "deleted"}
''',
"backend/Dockerfile": '''FROM python:3.10-slim

RUN apt-get update && apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 wget && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p weights && wget -q -O weights/realesr-general-x4v3.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth && wget -q -O weights/RealESRGAN_x4plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth && wget -q -O weights/RealESRGAN_x2plus.pth https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

COPY . .
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
''',
"backend/.env": '''API_SECRET_KEY=amed-super-secret-key-change-me-2025
MAX_FILE_SIZE_MB=25
MAX_INPUT_PIXELS=4000000
FREE_TIER_LIMIT=5
FILE_TTL_SECONDS=3600
'''
}
for path, content in FILES.items():
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"OK {path}")
print("Fait !")
