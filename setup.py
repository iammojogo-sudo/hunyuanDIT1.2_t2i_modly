#!/usr/bin/env python3
"""
HunyuanDiT v1.2 Text-to-Image — Modly extension setup script.

Called by Modly at install time:
    python setup.py <json_args>

json_args keys:
    python_exe  - path to Modly's embedded Python
    ext_dir     - absolute path to this extension directory
    gpu_sm      - GPU compute capability as integer (e.g. 89 for RTX 4090)
"""

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


IS_WIN = platform.system() == "Windows"


def python_exe_in_venv(venv):
    return venv / ("Scripts/python.exe" if IS_WIN else "bin/python")


def pip(venv, *args):
    venv_python = python_exe_in_venv(venv)
    subprocess.run([str(venv_python), "-m", "pip"] + list(args), check=True)


def setup(python_exe, ext_dir, gpu_sm):
    venv = ext_dir / "venv"

    print("[setup] Creating venv at %s ..." % venv)
    subprocess.run([str(python_exe), "-m", "venv", str(venv)], check=True)

    venv_python = python_exe_in_venv(venv)

    # ------------------------------------------------------------------ #
    # PyTorch — index chosen by GPU compute capability
    # ------------------------------------------------------------------ #
    if gpu_sm >= 100:
        torch_index = "https://download.pytorch.org/whl/cu128"
        torch_pkgs  = ["torch>=2.7.0", "torchvision>=0.22.0"]
        print("[setup] SM %d (Blackwell) -> PyTorch 2.7 + CUDA 12.8" % gpu_sm)
    elif gpu_sm >= 70:
        torch_index = "https://download.pytorch.org/whl/cu124"
        torch_pkgs  = ["torch==2.6.0", "torchvision==0.21.0"]
        print("[setup] SM %d -> PyTorch 2.6.0 + CUDA 12.4" % gpu_sm)
    else:
        torch_index = "https://download.pytorch.org/whl/cu118"
        torch_pkgs  = ["torch==2.5.1", "torchvision==0.20.1"]
        print("[setup] SM %d (legacy) -> PyTorch 2.5.1 + CUDA 11.8" % gpu_sm)

    print("[setup] Installing PyTorch ...")
    pip(venv, "install", *torch_pkgs, "--index-url", torch_index)

    # ------------------------------------------------------------------ #
    # Core dependencies
    # ------------------------------------------------------------------ #
    print("[setup] Installing core dependencies ...")
    pip(venv, "install",
        "diffusers>=0.29.0",
        "transformers>=4.40.0",
        "accelerate>=0.30.0",
        "huggingface_hub>=0.20.0",
        "sentencepiece",
        "safetensors",
        "Pillow",
        "numpy",
        "tiktoken",
        "protobuf",
    )

    # ------------------------------------------------------------------ #
    # Optional: xformers for memory-efficient attention
    # ------------------------------------------------------------------ #
    print("[setup] Installing xformers (optional) ...")
    try:
        if gpu_sm >= 70:
            pip(venv, "install", "xformers==0.0.29.post3",
                "--index-url", torch_index)
        else:
            pip(venv, "install", "xformers==0.0.28.post2",
                "--index-url", "https://download.pytorch.org/whl/cu118")
    except subprocess.CalledProcessError:
        print("[setup] xformers install failed — skipping (non-fatal).")

    print("[setup] Done. Venv ready at: %s" % venv)


if __name__ == "__main__":
    if len(sys.argv) >= 4:
        setup(
            python_exe=Path(sys.argv[1]),
            ext_dir=Path(sys.argv[2]),
            gpu_sm=int(sys.argv[3]),
        )
    elif len(sys.argv) == 2:
        args = json.loads(sys.argv[1])
        setup(
            python_exe=Path(args["python_exe"]),
            ext_dir=Path(args["ext_dir"]),
            gpu_sm=int(args["gpu_sm"]),
        )
    else:
        print("Usage: python setup.py <python_exe> <ext_dir> <gpu_sm>")
        print('   or: python setup.py \'{"python_exe":"...","ext_dir":"...","gpu_sm":89}\'')
        sys.exit(1)
