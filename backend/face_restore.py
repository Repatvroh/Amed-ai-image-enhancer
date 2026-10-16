import io
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
