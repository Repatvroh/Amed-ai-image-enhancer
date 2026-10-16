import numpy as np
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
