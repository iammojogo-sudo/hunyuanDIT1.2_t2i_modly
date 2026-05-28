"""
generator_worker.py — standalone worker spawned by processor.js.

Args: prompt  params_json  models_dir  workspace_dir

Communicates via newline-delimited JSON on stdout.
All other prints go to stderr (forwarded to context.log by processor.js).
"""

import io
import json
import os
import random
import sys
import time
import uuid
from pathlib import Path


def send(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def log(message: str) -> None:
    send({"type": "log", "message": message})


def progress(pct: int, step: str = "") -> None:
    send({"type": "progress", "pct": pct, "step": step})


def main() -> None:
    if len(sys.argv) < 5:
        send({"type": "error", "message": "Usage: generator_worker.py <prompt> <params_json> <models_dir> <workspace_dir>"})
        sys.exit(1)

    prompt        = sys.argv[1]
    params        = json.loads(sys.argv[2]) if sys.argv[2] else {}
    models_dir    = Path(sys.argv[3])
    workspace_dir = Path(sys.argv[4])

    model_dir = models_dir / "hunyuandit_1_2_t2i" / "generate"

    negative_prompt = params.get("negative_prompt") or None
    width           = int(params.get("width",  1024))
    height          = int(params.get("height", 1024))
    steps           = int(params.get("steps",  25))
    guidance_scale  = float(params.get("guidance_scale", 6.0))
    seed_val        = int(params.get("seed", 0))
    if seed_val == 0:
        seed_val = random.randint(1, 2 ** 31 - 1)

    progress(3, "Checking model weights...")

    if not model_dir.exists() or not (model_dir / "model_index.json").exists():
        send({"type": "error", "message": f"Model weights not found at {model_dir}. Download them from the Extensions page first."})
        sys.exit(1)

    progress(5, "Loading pipeline...")
    log(f"Loading HunyuanDiT from {model_dir}")

    try:
        import torch
        from diffusers import HunyuanDiTPipeline
    except ImportError as e:
        send({"type": "error", "message": f"Import failed: {e}. Run Repair on the extension."})
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.float16 if device == "cuda" else torch.float32

    try:
        pipe = HunyuanDiTPipeline.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=dtype,
        ).to(device)
    except Exception as e:
        send({"type": "error", "message": f"Failed to load pipeline: {e}"})
        sys.exit(1)

    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass

    progress(15, f"Generating on {device}...")
    log(f"Prompt: {prompt!r}  seed={seed_val}  {width}x{height}  steps={steps}")

    generator = torch.Generator(device=device).manual_seed(seed_val)

    try:
        with torch.inference_mode():
            out = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
        image = out.images[0]
    except Exception as e:
        send({"type": "error", "message": f"Generation failed: {e}"})
        sys.exit(1)

    progress(95, "Saving image...")

    out_dir = workspace_dir / "Workflows"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = "hunyuandit_%d_%s.png" % (int(time.time()), uuid.uuid4().hex[:8])
    out_path = out_dir / out_name

    try:
        image.save(str(out_path), format="PNG")
    except Exception as e:
        send({"type": "error", "message": f"Failed to save image: {e}"})
        sys.exit(1)

    progress(100, "Done")
    log(f"Saved to {out_path}")
    send({"type": "done", "output_path": str(out_path)})


if __name__ == "__main__":
    main()
