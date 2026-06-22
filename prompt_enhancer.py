import re
import sys

# keep stdout clean for the runner protocol (same convention as generator.py)
_print = print


def print(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    _print(*args, **kwargs)


# --- swap these to change the enhancer ---------------------------------------
# Qwen/Qwen3-1.7B   -> default, good quality, ~3.5GB, strong Chinese (matches HunyuanDiT)
# Qwen/Qwen3-0.6B   -> lighter / faster on CPU, smaller download
# Qwen/Qwen3-4B-Instruct-2507 -> best quality, heavier
ENHANCER_MODEL = "Qwen/Qwen3-1.7B"

# "cpu" keeps the image model's 6GB VRAM budget untouched. Set "cuda" only if you
# have enough VRAM for both the LLM and the diffusion model at the same time.
ENHANCER_DEVICE = "cpu"

MAX_NEW_TOKENS = 80

# HunyuanDiT's CLIP encoder only reads the first 77 tokens. Capping the output
# at ~40 words keeps the whole prompt inside that window (no silent truncation).
MAX_PROMPT_WORDS = 40
# -----------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a prompt engineer for the HunyuanDiT text-to-image model. "
    "Expand the user's short description into one vivid image prompt. "
    "Add concrete visual detail: subject, setting, lighting, mood, composition, "
    "color palette, and art style, while keeping the user's original subject and intent. "
    "Write in the same language as the user's input. "
    "Keep it under 40 words and front-load the most important details - the model's "
    "CLIP encoder only reads the first 77 tokens, so a tight prompt works better. "
    "Respond with only the final prompt as a single paragraph - no preamble, "
    "no quotation marks, no lists, and no explanation."
)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


class PromptEnhancer(object):
    def __init__(self, model_id=None, device=None):
        self.model_id = model_id or ENHANCER_MODEL
        self.device = device or ENHANCER_DEVICE
        self._tok = None
        self._model = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device == "cuda" and not torch.cuda.is_available():
            print("[enhancer] cuda requested but not available, falling back to cpu")
            self.device = "cpu"

        print("[enhancer] loading %s on %s" % (self.model_id, self.device))
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=dtype)
        self._model = model.to(self.device)
        self._model.eval()
        print("[enhancer] ready")

    def enhance(self, prompt):
        prompt = (prompt or "").strip()
        if not prompt:
            return prompt

        import torch

        self._ensure_loaded()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Qwen3 hybrid models default to "thinking" mode; turn it off for a clean,
        # single-paragraph result. The -2507 Instruct variant ignores this kwarg.
        try:
            text = self._tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = self._tok.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = self._tok(text, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tok.eos_token_id,
            )

        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        result = self._tok.decode(new_tokens, skip_special_tokens=True)
        return self._clean(result, fallback=prompt)

    def _clean(self, text, fallback):
        text = _THINK_RE.sub("", text or "")
        text = text.strip().strip('"').strip("'").strip()
        text = " ".join(text.split())
        if not text:
            return fallback
        return self._cap_words(text)

    def _cap_words(self, text):
        words = text.split()
        if len(words) <= MAX_PROMPT_WORDS:
            return text
        words = words[:MAX_PROMPT_WORDS]
        # drop a dangling connector left by the cut so it doesn't end on "and"/"with"
        while words and words[-1].lower().strip(",.;") in ("and", "with", "in", "of", "a", "the"):
            words.pop()
        return " ".join(words).rstrip(",;. ")

    def unload(self):
        self._model = None
        self._tok = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


_SINGLETON = None


def get_enhancer(model_id=None, device=None):
    global _SINGLETON
    changed = _SINGLETON is not None and (
        (device is not None and _SINGLETON.device != device)
        or (model_id is not None and _SINGLETON.model_id != model_id)
    )
    if _SINGLETON is None or changed:
        if _SINGLETON is not None:
            _SINGLETON.unload()
        _SINGLETON = PromptEnhancer(model_id=model_id, device=device)
    return _SINGLETON
