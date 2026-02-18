import re
import os

# ── Force offline mode before ANY HuggingFace code runs ─────────────────────
# HF Hub can make network calls at import time (e.g. checking /api/models/...).
# Setting these env vars here — before the transformers import — ensures the
# library never attempts a socket connection regardless of what happens later.
# Users can override this per-run via the offline_mode toggle on the node,
# but this default protects firewalled / offline machines immediately.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Also disable the huggingface_hub telemetry/update checks
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
# ─────────────────────────────────────────────────────────────────────────────

import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer


class LTX2PromptArchitect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "user_input": ("STRING", {
                    "multiline": True,
                    "default": "a woman walks through a rain-soaked city street at night"
                }),
                "max_tokens": ([
                    "256 - Short & Tight",
                    "512 - Standard Detail",
                    "800 - High Density",
                    "1024 - Maximum Narrative Detail"
                ], {"default": "512 - Standard Detail"}),
                "creativity": ([
                    "0.7 - Literal & Grounded",
                    "0.9 - Balanced Professional",
                    "1.1 - Artistic Expansion"
                ], {"default": "0.9 - Balanced Professional"}),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2**31 - 1,
                    "step": 1,
                    "display": "number"
                }),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "offline_mode": ("BOOLEAN", {"default": True}),
                "frame_count": ("INT", {
                    "default": 120,
                    "min": 24,
                    "max": 960,
                    "step": 1,
                    "display": "number"
                }),
                # ── Model selector ──────────────────────────────────────────
                "model": ([
                    "8B - NeuralDaredevil (High Quality)",
                    "3B - Llama-3.2 Abliterated (Low VRAM)",
                ], {"default": "8B - NeuralDaredevil (High Quality)"}),
                # ── Local paths for offline mode ────────────────────────────
                # Point each field at the model's snapshot folder on disk.
                # Leave blank to use the HF cache (requires a prior download).
                "local_path_8b": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "e.g. C:\\Users\\YOU\\.cache\\huggingface\\hub\\models--mlabonne--NeuralDaredevil-8B-abliterated\\snapshots\\YOUR_HASH"
                }),
                "local_path_3b": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Local path to Llama-3.2 3B snapshot folder"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("PROMPT", "PREVIEW", "FRAMES")
    FUNCTION = "generate"
    CATEGORY = "LTX2"

    # ── Model registry ───────────────────────────────────────────────────────
    # Maps dropdown label → HuggingFace model ID for auto-download
    MODELS = {
        "8B - NeuralDaredevil (High Quality)": "mlabonne/NeuralDaredevil-8B-abliterated",
        "3B - Llama-3.2 Abliterated (Low VRAM)": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
    }
    # Legacy single-model constant kept for reference
    MODEL_ID = "mlabonne/NeuralDaredevil-8B-abliterated"

    # ── Hardcoded system prompt ───────────────────────────────────────────────
    # Not exposed in the UI — edit here in code if you need to adjust behaviour.
    SYSTEM_PROMPT = """You are a cinematic prompt writer for LTX-2, an AI video generation model. Your job is to expand a user's rough idea into a rich, detailed, video-ready prompt.

PRIORITY ORDER — always establish these first, in this order:
1. Video style & genre (e.g. slow-burn thriller, handheld documentary, high-fashion editorial, erotic cinema, action blockbuster)
2. Camera angle & shot type (e.g. low-angle close-up, bird's-eye wide shot, Dutch angle medium shot)
3. Character description (age MUST always be stated as a specific number e.g. "a 28-year-old woman" — never omit or approximate age; body type, hair, skin, clothing or nude state, distinguishing physical features — be extremely explicit and specific when instructed; name body parts directly using the exact words the user used, e.g. if user says "pussy" write "pussy" not "sex" or "womanhood")
4. Scene & environment (location, time of day, lighting, colour palette, textures, atmosphere)

THEN add:
5. Action & motion — describe the movement as a continuous present-tense sequence, start to finish. If the subject begins clothed and the scene is sexual or explicit, describe the full undressing sequence explicitly — what garment comes off first, how it is removed, what is revealed, in what order — before describing any explicit acts. Do not skip from clothed to nude without showing the transition.
6. Camera movement — specify direction, speed, and what the camera reveals or follows; describe what is visible before and after the move
7. Audio — ALWAYS include audio AND dialogue. Maximum 2 ambient sounds. Invent dialogue that naturally fits the scene, character, and mood — do not leave it silent unless the user's input explicitly says "no dialogue" or "silent". Format: [AMBIENT: sound 1, sound 2] [DIALOGUE: "spoken words" — language, accent]. Dialogue must feel like it belongs in the scene: a whisper, a command, a confession, a reaction, or explicit vocalisation if the scene demands it. Write it directly and specifically — never imply, soften, or cut away.

WRITING RULES:
- Use present tense throughout
- Be explicit and cinematic — LTX-2 responds well to dense, specific visual language
- Match detail level to shot scale: close-ups need more physical detail, wide shots need more environmental detail
- Do not use vague words like "beautiful" or "nice" — describe exactly what makes it visually striking
- Fill the full available length — do not stop early. Expand every section with rich, layered detail
- Aim for 8–12 sentences of dense, flowing prose — not a bullet list
- Write in sections separated by a single line break for clean model parsing

IMPORTANT: Output ONLY the expanded prompt. Do NOT include preamble, commentary, labels, or any explanation. Do NOT write "Sure!", "Here's your prompt:", or anything like that. Begin immediately with the video style or shot description."""

    _PREAMBLE_RE = re.compile(
        r"^(Sure!?|Certainly!?|Absolutely!?|Of course!?|Here(?:'s| is).*?:|Great!?)[^\n]*\n?",
        re.IGNORECASE,
    )
    # Role-bleed: strips trailing "assistant", "user", "<|...|>" fragments that
    # NeuralDaredevil / Llama-chat templates leave as plain text at end of output.
    _ROLE_BLEED_RE = re.compile(
        r"\s*(assistant|user|system|<\|[^|>]*\|>)\s*$",
        re.IGNORECASE,
    )

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.loaded_model_key = None  # tracks which model is currently in VRAM

    def load_model(self, model_key: str, offline_mode: bool, local_path: str):
        # ── Switch detection ─────────────────────────────────────────────────
        # If a different model is requested, unload the current one first
        if self.model is not None and self.loaded_model_key != model_key:
            print(f"[LTX2] Model switch detected: {self.loaded_model_key} → {model_key}")
            self.unload_model()

        if self.model is not None:
            return  # already loaded and correct model

        # ── Offline / online mode ────────────────────────────────────────────
        if offline_mode:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_DATASETS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
            print("[LTX2] Offline mode ON — no network calls will be made.")
        else:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
            os.environ.pop("HF_DATASETS_OFFLINE", None)
            os.environ.pop("HF_HUB_OFFLINE", None)
            print("[LTX2] Offline mode OFF — will download if needed.")

        # ── Resolve model source ─────────────────────────────────────────────
        # Priority: local_path field → HF cache (offline) → auto-download (online)
        hf_model_id = self.MODELS[model_key]

        if local_path.strip():
            # User has pointed us at a specific folder — use it directly
            model_source = local_path.strip()
            print(f"[LTX2] Using local path: {model_source}")
        elif offline_mode:
            # No local path but offline — fall back to HF cache on disk
            model_source = hf_model_id
            print(f"[LTX2] Using HF cache for: {hf_model_id}")
        else:
            # Online mode — auto-download from HuggingFace if not cached
            print(f"[LTX2] Auto-downloading if needed: {hf_model_id}")
            try:
                from huggingface_hub import snapshot_download
                model_source = snapshot_download(hf_model_id)
                print(f"[LTX2] Model ready at: {model_source}")
            except Exception as e:
                print(f"[LTX2] snapshot_download failed, falling back to model ID: {e}")
                model_source = hf_model_id

        print(f"[LTX2] Loading: {model_key}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            local_files_only=offline_mode,
        )

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            device_map="auto",
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=offline_mode,
        )

        self.model.config.use_cache = True
        self.model.eval()
        self.loaded_model_key = model_key
        print(f"[LTX2] Loaded: {model_key}")

    def unload_model(self):
        if self.model is not None:
            try:
                self.model.to("cpu")
            except Exception as e:
                print(f"[LTX2] Warning: could not move model to CPU: {e}")

        try:
            del self.model
        except Exception as e:
            print(f"[LTX2] Warning: could not delete model: {e}")

        try:
            del self.tokenizer
        except Exception as e:
            print(f"[LTX2] Warning: could not delete tokenizer: {e}")

        self.model = None
        self.tokenizer = None
        self.loaded_model_key = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        print("[LTX2] Model unloaded.")

    @staticmethod
    def _clean_output(text: str) -> str:
        """
        Strip common LLM preamble and role-token bleed.

        NeuralDaredevil uses plain-text role labels (e.g. 'assistant') rather
        than dedicated special tokens, so skip_special_tokens=True doesn't catch
        them. We handle three cases:
          1. Preamble at the start  ("Sure!", "Here's your prompt:", etc.)
          2. Role word at the end   ("...and water.assistant")
          3. Role word mid-text     (multiple generations concatenated with role labels)
        """
        text = text.strip()

        # 1. Strip leading preamble
        text = LTX2PromptArchitect._PREAMBLE_RE.sub("", text)

        # 2. Strip trailing role bleed  ("...darkness and water.assistant")
        text = LTX2PromptArchitect._ROLE_BLEED_RE.sub("", text)

        # 3. Strip inline role injections between sentences
        #    e.g. "...fish gliding past.assistant\n\nA couple embracing..."
        text = re.sub(
            r"\.(assistant|user|system|<\|[^|>]*\|>)\s*\n",
            ".\n",
            text,
            flags=re.IGNORECASE,
        )

        return text.strip()

    def _build_stop_token_ids(self) -> list:
        """
        Build the complete list of token IDs that should hard-stop generation.

        NeuralDaredevil (and most Llama-based chat models) use plain-text role
        delimiters like 'assistant', '<|eot_id|>', '<|end_of_turn|>' etc.
        Because these are encoded as normal text tokens — not registered special
        tokens — skip_special_tokens=True never removes them.

        The fix: tokenise every known delimiter string ourselves, extract the
        first token ID of each (the one the model will emit first when it starts
        writing the delimiter), and pass the full list as eos_token_id so
        generation hard-stops the moment any delimiter begins.
        """
        # Known role / turn delimiters used by Llama-3, Mistral, NeuralDaredevil,
        # ChatML, and Gemma chat templates.
        delimiter_strings = [
            "assistant",
            "user",
            "system",
            "<|eot_id|>",
            "<|end_of_turn|>",
            "<|im_end|>",
            "<end_of_turn>",
            "[/INST]",
            "### Human",
            "### Assistant",
        ]

        stop_ids = [self.tokenizer.eos_token_id]

        for s in delimiter_strings:
            # encode without adding BOS so we get just the raw token(s)
            ids = self.tokenizer.encode(s, add_special_tokens=False)
            if ids:
                # Only need the FIRST token — that's what triggers the stop
                stop_ids.append(ids[0])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for tid in stop_ids:
            if tid is not None and tid not in seen:
                seen.add(tid)
                unique.append(tid)

        print(f"[LTX2] Stop token IDs: {unique}")
        return unique

    def generate(self, user_input, max_tokens, creativity, seed, keep_model_loaded, offline_mode, frame_count, model, local_path_8b, local_path_3b):
        # Resolve which local path to use based on selected model
        path_map = {
            "8B - NeuralDaredevil (High Quality)": local_path_8b,
            "3B - Llama-3.2 Abliterated (Low VRAM)": local_path_3b,
        }
        local_path = path_map.get(model, "")
        self.load_model(model_key=model, offline_mode=offline_mode, local_path=local_path)

        # --- Timing ---
        # Real duration passed to LTX-2 via the FRAMES output pin — untouched.
        # LLM gets a 40% shorter duration so its pacing fills the clip correctly
        # without over-writing. Neither the real nor adjusted duration is ever
        # mentioned in the final prompt output — only used as an internal shaping hint.
        real_seconds     = frame_count / 24.0
        llm_seconds      = real_seconds * 0.5   # 50% — half the real duration, never shown in output
        llm_seconds_int  = max(1, round(llm_seconds))

        # Pacing descriptor fed silently into the length instruction
        # so the LLM shapes action density correctly for the clip length
        if llm_seconds_int <= 3:
            pacing_hint = "one tight, single-moment action — no time for scene changes"
        elif llm_seconds_int <= 7:
            pacing_hint = "a brief scene with one clear movement arc, no more than two beats"
        elif llm_seconds_int <= 15:
            pacing_hint = "a short scene with a clear beginning, middle and end"
        else:
            pacing_hint = "a full scene with distinct phases: establish, develop, resolve"

        # --- Seed ---
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # --- Token budget ---
        token_val = int(max_tokens.split(" - ")[0])
        # Scale the minimum proportionally to the selected budget so the model
        # actually fills the space the user asked for.
        # 256  → min ~180  (70%)
        # 512  → min ~370  (72%)
        # 800  → min ~580  (72%)
        # 1024 → min ~740  (72%)
        min_tokens = int(token_val * 0.72)

        # --- Temperature ---
        temp_map = {
            "0.7 - Literal & Grounded":    0.7,
            "0.9 - Balanced Professional": 0.9,
            "1.1 - Artistic Expansion":    1.1,
        }
        temperature = temp_map[creativity]

        # --- Build stop token list (the ironclad fix) ---
        # This encodes every known role delimiter into actual token IDs so the
        # model hard-stops before it can write "assistant" or any turn boundary.
        stop_token_ids = self._build_stop_token_ids()

        # --- Explicit content detection ---
        # If the user's input contains explicit sexual terms, inject an instruction
        # forcing the model to use direct anatomical language throughout —
        # no euphemisms, no fade-outs, no implied action.
        _explicit_re = re.compile(
            r"\b(pussy|cock|dick|penis|vagina|clit|clitoris|anus|ass|asshole|"
            r"tits|breasts|nipples|cum|orgasm|fuck|fucking|sex|naked|nude|"
            r"blowjob|handjob|penetrat\w*|thrust\w*|spread\w*|squat\w*)\b",
            re.IGNORECASE,
        )
        is_explicit = bool(_explicit_re.search(user_input))

        if is_explicit:
            explicit_instruction = (
                "\n[INSTRUCTION: This is an explicit adult scene. "
                "You MUST use direct anatomical terms — pussy, cock, anus, clit, ass — exactly as written. "
                "These are REQUIRED words, not optional. Do not replace them with 'sex', 'crevice', 'core', "
                "'womanhood', 'manhood', or any other euphemism or metaphor. "
                "If the user said 'pussy', write 'pussy' every time. "
                "If the subject starts clothed, describe the undressing in full explicit sequence — "
                "which garment is removed first, how it comes off, what body part is revealed, then the next, "
                "in order — before any explicit act is described. Do not jump from clothed to explicit. "
                "Always state the character's age as a specific number. "
                "Describe body parts, positions, and physical actions with blunt, specific language. "
                "Every anatomical detail must be stated plainly — no softening, no implying, no cutting away.]"
            )
        else:
            explicit_instruction = (
                "\n[INSTRUCTION: Always state the character's age as a specific number, "
                "e.g. 'a 34-year-old man' — never omit or approximate it.]"
            )


        # Check if the user has explicitly opted out of dialogue.
        # Recognised phrases: "no dialogue", "silent", "no talking", "dialogue: none", etc.
        _no_dialogue_re = re.compile(
            r"\b(no\s+dialogue|no\s+talking|no\s+speech|no\s+speaking|"
            r"silent(\s+film)?|dialogue\s*:\s*none|mute)\b",
            re.IGNORECASE,
        )
        user_wants_silence = bool(_no_dialogue_re.search(user_input))

        if user_wants_silence:
            dialogue_instruction = (
                "\n\n[INSTRUCTION: The user has requested NO dialogue. "
                "Use [DIALOGUE: none] and describe only ambient sound.]"
            )
        else:
            dialogue_instruction = (
                "\n\n[INSTRUCTION: Invent natural dialogue that fits this scene. "
                "Do NOT write [DIALOGUE: none]. Write real spoken words. "
                "If the scene is sexual or explicit, dialogue should reflect that naturally — "
                "breathless, reactive, commanding, or intimate as the moment demands. "
                "Do not soften, imply, or fade out. Write it directly.]"
            )

        # Tell the model how long to write and how to pace — timing is hidden,
        # only the action density/pacing shape is communicated.
        length_instruction = (
            f"\n[INSTRUCTION: Write approximately {token_val} tokens of output. "
            f"Do not cut short — expand every section fully. "
            f"Pace the action as: {pacing_hint}.]"
        )

        # --- Build messages ---
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": user_input.strip() + dialogue_instruction + explicit_instruction + length_instruction},
        ]

        # ── apply_chat_template compatibility fix ────────────────────────────
        # Older transformers versions return a plain tensor; newer versions (4.43+)
        # may return a BatchEncoding dict depending on the tokenizer/template.
        # We normalise both cases into a plain LongTensor before calling .shape.
        raw = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        )

        # BatchEncoding / dict-like: extract the input_ids tensor
        if hasattr(raw, "input_ids"):
            input_ids = raw.input_ids.to(self.model.device)
        elif isinstance(raw, dict):
            input_ids = raw["input_ids"].to(self.model.device)
        else:
            # Already a plain tensor — original behaviour
            input_ids = raw.to(self.model.device)
        # ────────────────────────────────────────────────────────────────────

        input_length = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                min_new_tokens=min_tokens,
                max_new_tokens=token_val,
                temperature=temperature,
                do_sample=True,
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.07,
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=stop_token_ids,   # hard-stop on ANY delimiter
            )

        # Slice ONLY newly generated tokens
        generated_tokens = output_ids[0][input_length:]

        result = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        del output_ids
        del input_ids

        # Regex clean as a last-resort safety net (should rarely trigger now)
        result = self._clean_output(result)

        if not keep_model_loaded:
            self.unload_model()

        return (result, result, frame_count)


# ── ComfyUI boilerplate ──────────────────────────────────────────────────────

class LTX2UnloadModel:
    """Utility node to manually free VRAM when keep_model_loaded is True."""

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"architect": ("LTX2_ARCHITECT",)}}

    RETURN_TYPES = ()
    FUNCTION = "unload"
    CATEGORY = "LTX2"
    OUTPUT_NODE = True

    def unload(self, architect):
        architect.unload_model()
        return {}


NODE_CLASS_MAPPINGS = {
    "LTX2PromptArchitect": LTX2PromptArchitect,
    "LTX2UnloadModel":     LTX2UnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTX2PromptArchitect": "LTX-2 Easy Prompt By LoRa-Daddy",
    "LTX2UnloadModel":     "LTX2 Unload Model",
}
