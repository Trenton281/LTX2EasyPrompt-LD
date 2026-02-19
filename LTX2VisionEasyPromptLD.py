import gc
import os
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


def comfy_tensor_to_pil(tensor) -> Image.Image:
    if tensor.ndim == 4:
        tensor = tensor[0]
    arr = (tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


# One shared instance so VRAM is reused if node runs multiple times
_INSTANCE = {"processor": None, "model": None, "source": None}

DESCRIBE_PROMPT = (
    "Describe this image in one paragraph of plain sentences, around 100-130 words. "
    "Start with 'Style: photorealistic' or 'Style: anime' or 'Style: 3D animation' etc. "
    "Then describe the person naturally — their age, gender, skin tone, hair, body type, "
    "what they are wearing or doing, and any exposed body parts you can see — "
    "use plain words: breasts, nipples, pussy, penis, anus, buttocks. "
    "Describe their pose, what they are on or interacting with, "
    "the camera framing and angle, the lighting and time of day, and the setting. "
    "Write it as one flowing paragraph. Do not use bullet points, lists, or labels. "
    "If there is no person in the image, describe the scene instead — the environment, setting, lighting, time of day, mood, and any notable objects or details."
)



MODEL_OPTIONS = {
    "Qwen2.5-VL-3B — Fast (huihui abliterated)":    "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
    "Qwen2.5-VL-7B — Better NSFW (prithiv caption)": "prithivMLmods/Qwen2.5-VL-7B-Abliterated-Caption-it",
}

VISION_SYSTEM_PROMPT = (
    "You are an image description tool for an AI video pipeline. "
    "Describe exactly what you see in plain factual prose. "
    "Be direct and accurate. Do not embellish or invent details."
)


# ── OpenAI-compatible API helpers for vision ──────────────────────────────────
# Reuses the same pattern as LTX2EasyPromptLD.py but adds base64 image support.

_VISION_API_MODELS_CACHE: list = ["(server unreachable — check host/port)"]


def _get_vision_api_models_list() -> list:
    """Fetch models from default LM Studio address for the dropdown."""
    _fetch_vision_api_models("127.0.0.1", 1234, timeout=2)
    return _VISION_API_MODELS_CACHE[:]


def _fetch_vision_api_models(host: str, port: int, timeout: int = 5) -> list:
    """GET /v1/models from the API server. Updates cache in-place."""
    import urllib.request
    import json

    global _VISION_API_MODELS_CACHE
    url = f"http://{host}:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m["id"] for m in data.get("data", []) if "id" in m]
            _VISION_API_MODELS_CACHE = models if models else ["(no models loaded on server)"]
    except Exception as e:
        print(f"[VisionDescribe-API] /v1/models fetch failed ({url}): {e}")
        _VISION_API_MODELS_CACHE = ["(server unreachable — check host/port)"]
    return _VISION_API_MODELS_CACHE[:]


def _pil_to_base64(pil_image: Image.Image) -> str:
    """Convert a PIL image to a base64-encoded JPEG string for the API."""
    import base64
    import io

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _call_vision_api(host: str, port: int, messages: list, params: dict,
                     timeout: int = 120) -> str:
    """
    POST /v1/chat/completions with vision content (base64 image).
    Returns the assistant message content string.
    """
    import urllib.request
    import urllib.error
    import json

    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model":       params["model"],
        "messages":    messages,
        "temperature": params.get("temperature", 0.3),
        "max_tokens":  params.get("max_tokens", 300),
        "stream":      False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type":  "application/json",
            "Authorization": "Bearer local",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"[VisionDescribe-API] HTTP {e.code} from {url}: {err_body}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"[VisionDescribe-API] Cannot connect to {url}. "
            f"Is LM Studio running? ({e.reason})"
        ) from e
    except (KeyError, IndexError, Exception) as e:
        raise RuntimeError(f"[VisionDescribe-API] Unexpected response format: {e}") from e


def _unload_vision_api_model(host: str, port: int, model_id: str) -> None:
    """LM Studio-specific: POST /api/v1/models/unload to free VRAM."""
    import urllib.request
    import json

    url = f"http://{host}:{port}/api/v1/models/unload"
    payload = json.dumps({"instance_id": model_id}).encode("utf-8")
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as _:
            print(f"[VisionDescribe-API] Model unloaded from server: {model_id}")
    except Exception as e:
        print(f"[VisionDescribe-API] Model unload not supported by this server (OK): {e}")


class LTX2VisionDescribe:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Connect your starting image here. The vision model will analyse it and output a scene description for use with the Easy Prompt node."}),
                "backend": (
                    ["Transformers (Direct)", "OpenAI-Compatible API"],
                    {"default": "Transformers (Direct)"},
                ),
                "model": (list(MODEL_OPTIONS.keys()), {
                    "default": "Qwen2.5-VL-3B — Fast (huihui abliterated)",
                    "tooltip": "3B is faster and uses ~6GB VRAM. 7B is slower but describes explicit content more accurately. Both download automatically on first run."
                }),
                "offline_mode": ("BOOLEAN", {"default": False, "tooltip": "Turn ON if you have no internet connection. Uses locally cached models only. Leave OFF to allow automatic download from HuggingFace on first run."}),
                "local_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional: local snapshot path (overrides model dropdown)",
                    "tooltip": "Optional. Paste the full path to a locally downloaded model snapshot folder. This overrides the model dropdown above. Leave blank to use HuggingFace cache automatically."
                }),
                "gpu_memory_gb": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 80,
                    "step": 1,
                    "display": "number",
                    "tooltip": "Maximum GPU VRAM to use in GB. Set to 0 to use all available VRAM (default). If you get OOM errors with the 7B, set this to e.g. 10 and the overflow spills to system RAM automatically."
                }),
                # ── API backend settings (ignored when backend = Transformers) ──
                "api_host": ("STRING", {
                    "default": "127.0.0.1",
                    "multiline": False,
                    "placeholder": "LM Studio server address",
                }),
                "api_port": ("INT", {
                    "default": 1234,
                    "min": 1,
                    "max": 65535,
                    "step": 1,
                    "display": "number",
                }),
                "api_model": (_get_vision_api_models_list(), {
                    "default": _get_vision_api_models_list()[0],
                }),
                "api_model_custom": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Override: type model ID directly (use if dropdown is empty)",
                }),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("scene_context",)
    FUNCTION      = "describe"
    CATEGORY      = "LTX2"

    def describe(self, image, backend="Transformers (Direct)", model="",
                 offline_mode=True, local_path="",
                 gpu_memory_gb=0,
                 api_host="127.0.0.1", api_port=1234,
                 api_model="", api_model_custom=""):
        global _INSTANCE

        # ── API backend route ─────────────────────────────────────────────────
        if backend == "OpenAI-Compatible API":
            return self._describe_api(
                image=image,
                host=api_host,
                port=api_port,
                model=api_model_custom.strip() or api_model,
            )

        # ── Original transformers backend (Sean's code, unchanged) ────────────
        hf_id = MODEL_OPTIONS[model]

        # ── Offline env ───────────────────────────────────────────────────────
        if offline_mode:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
        else:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)

        # ── Resolve source ────────────────────────────────────────────────────
        if local_path and local_path.strip():
            source = local_path.strip()
        elif offline_mode:
            source = hf_id
        else:
            try:
                from huggingface_hub import snapshot_download
                source = snapshot_download(hf_id)
            except Exception as e:
                print(f"[VisionDescribe] Download failed: {e}")
                source = hf_id

        # ── Load if needed ────────────────────────────────────────────────────
        if _INSTANCE["model"] is None or _INSTANCE["source"] != source:
            # Clear any previous instance first
            if _INSTANCE["model"] is not None:
                try:
                    _INSTANCE["model"].to("cpu")
                except Exception:
                    pass
                _INSTANCE["model"]     = None
                _INSTANCE["processor"] = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            print(f"[VisionDescribe] Loading {model}...")

            # Build max_memory dict so accelerate caps GPU usage and spills to RAM.
            # If gpu_memory_gb is 0, let accelerate use all available VRAM automatically.
            if gpu_memory_gb > 0:
                max_memory = {0: f"{gpu_memory_gb}GiB", "cpu": "48GiB"}
                print(f"[VisionDescribe] GPU cap: {gpu_memory_gb}GB — overflow will spill to system RAM.")
            else:
                max_memory = None

            _INSTANCE["processor"] = AutoProcessor.from_pretrained(
                source, local_files_only=offline_mode
            )
            _INSTANCE["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                source,
                device_map="auto",
                torch_dtype=dtype,
                local_files_only=offline_mode,
                max_memory=max_memory,
            )
            _INSTANCE["model"].eval()
            _INSTANCE["source"] = source
            print("[VisionDescribe] Loaded.")

        processor = _INSTANCE["processor"]
        model     = _INSTANCE["model"]

        pil_image = comfy_tensor_to_pil(image)
        print(f"[VisionDescribe] Image: {pil_image.size}")

        # ── Single inference ──────────────────────────────────────────────────
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError("[VisionDescribe] Missing: qwen-vl-utils. Fix: pip install qwen-vl-utils then restart ComfyUI.")

        messages = [
            {
                "role": "system",
                "content": VISION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text",  "text":  DESCRIBE_PROMPT},
                ],
            },
        ]

        text_input = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        input_len = inputs["input_ids"].shape[1]

        tok = processor.tokenizer
        stop_ids = []
        if tok.eos_token_id is not None:
            stop_ids.append(tok.eos_token_id)
        for s in ["<|im_end|>", "<|endoftext|>"]:
            ids = tok.encode(s, add_special_tokens=False)
            if len(ids) == 1 and ids[0] not in stop_ids:
                stop_ids.append(ids[0])

        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=180,   # enough for ~100 words with some headroom
                temperature=0.3,      # low temp = factual, consistent
                do_sample=True,
                top_p=0.9,
                pad_token_id=pad_id,
                eos_token_id=stop_ids,
            )

        new_tokens = out[0][input_len:]
        description = tok.decode(new_tokens, skip_special_tokens=True).strip()

        del out, inputs

        print(f"[VisionDescribe] Output: {len(description.split())} words.")

        # ── Unload immediately to free VRAM for the text node ─────────────────
        print("[VisionDescribe] Unloading to free VRAM...")
        try:
            _INSTANCE["model"].to("cpu")
        except Exception:
            pass
        _INSTANCE["model"]     = None
        _INSTANCE["processor"] = None
        _INSTANCE["source"]    = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("[VisionDescribe] VRAM cleared.")

        return (description,)

    def _describe_api(self, image, host, port, model):
        """
        Describe an image via an OpenAI-compatible API server (LM Studio).
        Sends the image as base64 JPEG in the chat completions request.
        Uses the same DESCRIBE_PROMPT as the transformers path.
        """
        # Refresh model list cache
        _fetch_vision_api_models(host, port)

        # Validate model selection
        effective_model = model.strip()
        if not effective_model or effective_model.startswith("("):
            return (
                "[VisionDescribe-API] No model selected. "
                "Type a model ID in 'api_model_custom' or ensure LM Studio is running "
                "and has a vision model loaded.",
            )

        # Convert ComfyUI image tensor to base64 JPEG
        pil_image = comfy_tensor_to_pil(image)
        print(f"[VisionDescribe-API] Image: {pil_image.size}")
        image_b64 = _pil_to_base64(pil_image)

        # Build messages with base64 image (OpenAI vision format)
        messages = [
            {
                "role": "system",
                "content": VISION_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": DESCRIBE_PROMPT,
                    },
                ],
            },
        ]

        try:
            description = _call_vision_api(
                host=host,
                port=port,
                messages=messages,
                params={
                    "model":       effective_model,
                    "temperature": 0.3,
                    "max_tokens":  300,
                },
            )
        except RuntimeError as e:
            error_str = str(e)
            print(error_str)
            return (error_str,)

        print(f"[VisionDescribe-API] Output: {len(description.split())} words.")

        # Unload model from LM Studio to free VRAM
        _unload_vision_api_model(host, port, effective_model)

        return (description,)


NODE_CLASS_MAPPINGS = {
    "LTX2VisionDescribe": LTX2VisionDescribe,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2VisionDescribe": "LTX-2 Vision Describe By LoRa-Daddy",
}
