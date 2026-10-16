import uuid, time, asyncio
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
