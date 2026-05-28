# HunyuanDiT v1.2 Text-to-Image — Modly Extension

Text-to-image generation using Tencent's HunyuanDiT v1.2 Distilled model. Runs locally on your GPU, supports both English and Chinese prompts, and outputs at up to 1280×1280.

Weights are ~6GB. You'll need around 6GB of VRAM.

---

## Installation

1. Open Modly and go to the **Extensions** tab
2. Click **Install from GitHub** and paste this repo URL
3. Wait for setup to finish — it installs PyTorch and the required packages into an isolated environment so nothing on your system gets touched
4. Once installed, click **Download** on the Generate Image node to grab the model weights from HuggingFace

---

## Usage

This extension works in the **Workflows** tab.

1. Drag an **Image** node onto the canvas and point it at `placeholder.png` (included in this repo — it's just a 1×1 pixel image, it won't affect your output)
2. Drag a **Generate Image** node onto the canvas and connect the Image node to it
3. Type your prompt in the **Prompt** field on the Generate Image node
4. Hit **Run**

The generated image saves to your Modly workspace under `Workflows/`.

If you want to chain into 3D generation, connect the output of Generate Image directly into a Hunyuan3D or similar mesh node.

---

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| Prompt | — | What you want to generate |
| Negative Prompt | — | What to avoid |
| Width | 1024 | 512–1280 |
| Height | 1024 | 512–1280 |
| Steps | 25 | Higher = slower but more detail |
| Guidance Scale | 6.0 | How closely it follows the prompt |
| Seed | 0 | 0 picks a random seed each run |

---

## Notes

- First generation takes longer while the model loads into VRAM. Subsequent runs in the same session are faster.
- HunyuanDiT handles Chinese prompts natively alongside English.
- The `placeholder.png` in this repo is required by Modly's workflow system — it has no effect on what gets generated.
