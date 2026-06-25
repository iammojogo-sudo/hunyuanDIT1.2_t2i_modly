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

1. Drag a **Generate Image** node onto the canvas
2. Type your prompt in the **Prompt** field on the node
3. Optionally enter a **Negative Prompt** to describe what you want to avoid
4. Adjust parameters like resolution, steps, and guidance scale as needed
5. Hit **Run**

The generated image saves to your Modly workspace under `Workflows/`.

---

## Parameters

| Parameter | Default | Notes |
|---|---|---|
| Prompt | — | What you want to generate |
| Enhance Prompt (LLM) | Disabled | Uses a local LLM to expand your prompt with more detail |
| Negative Prompt | — | What to avoid in the image |
| Width | 1024 | 512–1280 |
| Height | 1024 | 512–1280 |
| Steps | 25 | Higher = slower but more detail |
| Guidance Scale | 6.0 | How closely it follows the prompt |
| Seed | 0 | 0 picks a random seed each run |

---

## Notes

- First generation takes longer while the model loads into VRAM. Subsequent runs in the same session are faster.
- HunyuanDiT handles Chinese prompts natively alongside English.
- **v1.2.0**: Both positive and negative prompts are now entered directly as text fields on the node — no wire inputs or placeholder image required. A future update will restore wire-based text inputs once Modly's text-node support is merged into the main branch.

