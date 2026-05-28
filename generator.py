import random
import sys
import threading
import time
import uuid
from pathlib import Path

from services.generators.base import BaseGenerator, smooth_progress

_print = print
def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)

_HF_REPO_ID = "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled"

def _safe_int(val, default):
    try:    return int(val)
    except: return default

def _safe_float(val, default):
    try:    return float(val)
    except: return default


class HunyuanDiT12Generator(BaseGenerator):
    MODEL_ID     = "hunyuandit_1_2_t2i"
    DISPLAY_NAME = "HunyuanDiT v1.2 Text-to-Image"
    VRAM_GB      = 6

    def is_downloaded(self):
        if self.download_check:
            return (self.model_dir / self.download_check).exists()
        return (self.model_dir / "model_index.json").exists()

    def load(self):
        if self._model is not None:
            return
        if not self.is_downloaded():
            self._download_weights()

        import torch
        from diffusers import HunyuanDiTPipeline

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._dtype  = torch.float16 if self._device == "cuda" else torch.float32

        print("[HunyuanDiT] Loading from %s" % self.model_dir)
        pipe = HunyuanDiTPipeline.from_pretrained(
            str(self.model_dir),
            local_files_only=True,
            torch_dtype=self._dtype,
        ).to(self._device)
        try: pipe.set_progress_bar_config(disable=True)
        except: pass
        self._model = pipe
        print("[HunyuanDiT] Loaded on %s" % self._device)

    def unload(self):
        self._model  = None
        self._device = None
        self._dtype  = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except: pass

    def generate(self, image_bytes, params, progress_cb=None, cancel_event=None):
        import torch

        params = params or {}

        if self._model is None:
            self.load()

        prompt = params.get("prompt", "").strip()
        if not prompt:
            raise ValueError("prompt is required — type it in the Prompt field on the node.")

        negative_prompt = params.get("negative_prompt") or None
        width          = _safe_int(params.get("width"),  1024)
        height         = _safe_int(params.get("height"), 1024)
        steps          = _safe_int(params.get("steps"),  25)
        guidance_scale = _safe_float(params.get("guidance_scale"), 6.0)
        seed_val       = _safe_int(params.get("seed"), 0)
        if seed_val == 0:
            seed_val = random.randint(1, 2 ** 31 - 1)

        self._report(progress_cb, 5, "Starting generation ...")
        self._check_cancelled(cancel_event)

        generator = torch.Generator(device=self._device).manual_seed(seed_val)

        self._report(progress_cb, 10, "Generating image ...")
        stop_evt = threading.Event()
        t = None
        if progress_cb:
            t = threading.Thread(
                target=smooth_progress,
                args=(progress_cb, 10, 95, "Generating image ...", stop_evt),
                daemon=True,
            )
            t.start()

        try:
            with torch.inference_mode():
                out = self._model(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=width,
                    height=height,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
            image = out.images[0]
        finally:
            stop_evt.set()
            if t: t.join(timeout=1.0)

        self._check_cancelled(cancel_event)
        self._report(progress_cb, 98, "Saving ...")

        out_dir = self.outputs_dir if self.outputs_dir else (
            self.model_dir.parent.parent.parent / "outputs" / self.MODEL_ID
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = "hunyuandit_%d_%s.png" % (int(time.time()), uuid.uuid4().hex[:8])
        out_path = out_dir / out_name
        image.save(str(out_path), format="PNG")

        self._report(progress_cb, 100, "Done")
        print("[HunyuanDiT] Saved to %s" % out_path)
        return str(out_path)

    def _auto_download(self):
        self._download_weights()

    def _download_weights(self):
        from huggingface_hub import snapshot_download
        repo_id        = self.hf_repo or _HF_REPO_ID
        manifest_skips = list(getattr(self, "hf_skip_prefixes", []) or [])
        ignore = []
        for p in manifest_skips:
            ignore.append(p)
            if isinstance(p, str) and p.endswith("/"):
                ignore.append(p + "*")
        ignore += ["*.md", "*.txt", "LICENSE", "NOTICE", ".gitattributes"]
        self.model_dir.mkdir(parents=True, exist_ok=True)
        print("[HunyuanDiT] Downloading from %s ..." % repo_id)
        snapshot_download(repo_id=repo_id, local_dir=str(self.model_dir), ignore_patterns=ignore)
        print("[HunyuanDiT] Done.")
