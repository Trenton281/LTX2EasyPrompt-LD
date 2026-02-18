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


# ── Negative prompt builder ───────────────────────────────────────────────────
# Builds a scene-aware negative prompt without a second LLM call.
# Base quality terms are always included; scene-specific terms are added
# by scanning the generated prompt for relevant content.

_NEG_BASE = (
    "blurry, out of focus, low quality, worst quality, jpeg artifacts, "
    "static, no motion, frozen, duplicate, watermark, text, signature, "
    "poorly drawn, bad anatomy, deformed, disfigured, extra limbs, "
    "missing limbs, floating limbs, disconnected body parts, "
    "overexposed, underexposed, grainy, noise"
)

_NEG_INDOOR   = "harsh outdoor lighting, direct sunlight"
_NEG_OUTDOOR  = "studio background, indoor lighting"
_NEG_EXPLICIT = "censored, mosaic, pixelated, black bar, blurred genitals"
_NEG_PORTRAIT = "wide angle distortion, fish eye, full body shot"
_NEG_WIDE     = "close-up, portrait crop, tight frame"
_NEG_NIGHT    = "overexposed, bright daylight, blown highlights"
_NEG_DAY      = "underexposed, dark shadows, black crush"
_NEG_MULTI    = "merged bodies, fused figures, incorrect number of people"

def _build_negative_prompt(result: str, user_input: str) -> str:
    combined = (result + " " + user_input).lower()
    extras = []

    if any(w in combined for w in ["indoor", "room", "interior", "bedroom", "kitchen", "office"]):
        extras.append(_NEG_OUTDOOR)
    elif any(w in combined for w in ["outdoor", "street", "beach", "forest", "park", "exterior"]):
        extras.append(_NEG_INDOOR)

    if any(w in combined for w in ["pussy", "cock", "penis", "vagina", "nude", "naked", "explicit", "nipple", "breast"]):
        extras.append(_NEG_EXPLICIT)

    if any(w in combined for w in ["close-up", "close up", "portrait", "face shot", "headshot"]):
        extras.append(_NEG_PORTRAIT)
    elif any(w in combined for w in ["wide shot", "wide angle", "aerial", "bird's-eye", "establishing"]):
        extras.append(_NEG_WIDE)

    if any(w in combined for w in ["night", "dark", "moonlight", "dimly lit", "candlelight"]):
        extras.append(_NEG_NIGHT)
    elif any(w in combined for w in ["daylight", "sunny", "golden hour", "bright", "midday"]):
        extras.append(_NEG_DAY)

    if any(w in combined for w in ["two women", "two men", "two people", "both", "together", "couple", "they "]):
        extras.append(_NEG_MULTI)

    parts = [_NEG_BASE] + extras
    return ", ".join(parts)


class LTX2PromptArchitect:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bypass": ("BOOLEAN", {"default": False}),
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
                "invent_dialogue": ("BOOLEAN", {"default": True}),
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
            "optional": {
                "scene_context": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional: vision description from LTX-2 Vision Describe node",
                }),
                "lora_triggers": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Optional: LoRA trigger words e.g. 'ohwx woman, film grain'",
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("PROMPT", "PREVIEW", "NEG_PROMPT")
    FUNCTION = "generate"
    CATEGORY = "LTX2"

    # ── Model registry ───────────────────────────────────────────────────────
    # Maps dropdown label → HuggingFace model ID for auto-download
    MODELS = {
        "8B - NeuralDaredevil (High Quality)": "mlabonne/NeuralDaredevil-8B-abliterated",
        "3B - Llama-3.2 Abliterated (Low VRAM)": "huihui-ai/Llama-3.2-3B-Instruct-abliterated",
    }

    # ── Hardcoded system prompt ───────────────────────────────────────────────
    # Not exposed in the UI — edit here in code if you need to adjust behaviour.
    SYSTEM_PROMPT = """You are a cinematic prompt writer for LTX-2, an AI video generation model. Your job is to expand a user's rough idea into a rich, detailed, video-ready prompt.

PRIORITY ORDER — always establish these first, in this order:
1. Video style & genre (e.g. slow-burn thriller, handheld documentary, high-fashion editorial, erotic cinema, action blockbuster)
2. Camera angle & shot type (e.g. low-angle close-up, bird's-eye wide shot, Dutch angle medium shot)
3. Character description (age MUST always be stated as a specific number e.g. "a 28-year-old woman" — never omit or approximate age; body type, hair, skin, clothing or nude state, distinguishing physical features — be extremely explicit and specific when instructed; name body parts directly using the exact words the user used, e.g. if user says "pussy" write "pussy" not "sex" or "womanhood")
4. Scene & environment (location, time of day, lighting, colour palette, textures, atmosphere)

THEN add:
5. Action & motion — describe the movement as a continuous present-tense sequence, start to finish. Structure the action in clear beats.

UNDRESSING RULE — this is mandatory when clothing removal is implied or stated:
If the subject is clothed at the start and the scene involves nudity, stripping, or explicit content, you MUST dedicate a full narrative segment to the undressing BEFORE any nudity or explicit act is described. This segment must:
  - Name each garment explicitly (shirt, bra, jeans, underwear, shoes, etc.)
  - Describe HOW it is removed — fingers on buttons, zipper pulled down, fabric sliding off shoulders, stepped out of, peeled away
  - Describe what is physically REVEALED at each step — the collarbone, the stomach, the curve of the hip, the chest, the bare legs
  - Follow the exact order the clothes come off — do not skip or compress steps
  - Use camera-aware language — the camera lingers, cuts to, tilts down to reveal, holds on
  - Do NOT jump from clothed to naked. The transition IS the content. Treat it as its own scene segment with the same density as any other beat.

6. Camera movement — specify direction, speed, and what the camera reveals or follows; describe what is visible before and after the move
7. Audio — Always include ambient sound. Maximum 2 ambient sounds. Format: [AMBIENT: sound 1, sound 2]
   Dialogue — follow the DIALOGUE INSTRUCTION you are given exactly. When dialogue is included, write it as inline prose woven into the action — not as a labelled tag. The spoken words sit inside the sentence, attributed with delivery and physical action, exactly like a novel. Examples of correct format:
   'He leans back, satisfied, "I think I'll have to go back tomorrow for more," he chuckles, his eyes crinkling at the corners.'
   '"Don\'t stop," she breathes, gripping the sheets, her voice barely above a whisper.'
   'She turns to face him, "I\'ve been waiting all day for this," her tone quiet and certain.'
   NEVER use [DIALOGUE: ...] tags. NEVER write dialogue as a separate bracketed block. Dialogue flows inside the prose as part of the action.

WRITING RULES:
- Use present tense throughout
- Be explicit and cinematic — LTX-2 responds well to dense, specific visual language
- Match detail level to shot scale: close-ups need more physical detail, wide shots need more environmental detail
- Do not use vague words like "beautiful" or "nice" — describe exactly what makes it visually striking
- Fill the full available length — do not stop early. Expand every section with rich, layered detail
- Aim for 8–12 sentences of dense, flowing prose — not a bullet list
- Write in sections separated by a single line break for clean model parsing

IMPORTANT: Output ONLY the expanded prompt. Do NOT include preamble, commentary, labels, or any explanation. Do NOT write "Sure!", "Here's your prompt:", or anything like that. Do NOT add a checklist, compliance summary, note, or confirmation of instructions at the end — not in brackets, not as a "Note:", not in any form. The output ends when the scene ends. Nothing after the last sentence of the scene. Begin immediately with the video style or shot description."""

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
        Strip common LLM preamble, role-token bleed, and compliance checklists.

        NeuralDaredevil uses plain-text role labels (e.g. 'assistant') rather
        than dedicated special tokens, so skip_special_tokens=True doesn't catch
        them. We handle four cases:
          1. Preamble at the start  ("Sure!", "Here's your prompt:", etc.)
          2. Role word at the end   ("...and water.assistant")
          3. Role word mid-text     (multiple generations concatenated with role labels)
          4. Compliance checklist   ("(Exactly 4 actions...)(Pacing strict)..." etc.)
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

        # 4. Strip trailing compliance content — model sometimes appends:
        #    - A "Note:" explanation block after the scene ends
        #    - A single parenthesised summary line: "(5 distinct actions within 20 seconds)"
        #    - Consecutive bracketed phrases: "(Exactly 4 actions)(Pacing strict)..."
        #    Order matters: strip Note: first so it doesn't shield bracket lines above it.
        text = re.sub(
            r"\s*\n+Note:.*$",               # trailing Note: block (must go first)
            "",
            text,
            flags=re.DOTALL,
        ).strip()
        text = re.sub(
            r"(\s*\([^)]{5,}\)){2,}\s*$",   # consecutive bracketed phrases
            "",
            text,
        ).strip()
        text = re.sub(
            r"\s*\(\d+[^)]{3,}\)\s*$",       # single parenthesised count/summary line
            "",
            text,
        ).strip()

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

    def generate(self, bypass, user_input, max_tokens, creativity, seed, invent_dialogue, keep_model_loaded, offline_mode, frame_count, model, local_path_8b, local_path_3b, scene_context="", lora_triggers=""):
        # ── Bypass mode — no model loaded, input passed straight through ────────
        if bypass:
            print("[LTX2] Bypass ON — skipping model, passing user_input directly.")
            neg_prompt = _build_negative_prompt("", user_input)
            return (user_input.strip(), user_input.strip(), neg_prompt)

        # Resolve which local path to use based on selected model
        path_map = {
            "8B - NeuralDaredevil (High Quality)": local_path_8b,
            "3B - Llama-3.2 Abliterated (Low VRAM)": local_path_3b,
        }
        local_path = path_map.get(model, "")
        self.load_model(model_key=model, offline_mode=offline_mode, local_path=local_path)

        # --- Timing & pacing ---
        # Convert frames to real seconds, then calculate a hard action count cap.
        # One visible screen action takes roughly 4 seconds to read as distinct.
        # We clamp between 1 and 10 to stay sane at extremes.
        real_seconds = frame_count / 24.0
        action_count = max(1, min(10, round(real_seconds / 4)))

        # Build a concrete, number-based pacing instruction the LLM cannot fudge.
        # Vague descriptors like "short scene" get ignored — explicit counts don't.
        if action_count == 1:
            pacing_hint = (
                f"This clip is {real_seconds:.0f} seconds long. "
                f"Write EXACTLY 1 action. One single moment. "
                f"Do not describe anything before or after it. No setup, no resolution."
            )
        else:
            pacing_hint = (
                f"This clip is {real_seconds:.0f} seconds long. "
                f"Write EXACTLY {action_count} distinct actions — no more. "
                f"Each action should take roughly {real_seconds / action_count:.0f} seconds of screen time. "
                f"Do not add setup, backstory, or resolution beyond these {action_count} actions. "
                f"Stop when the {action_count}{'st' if action_count == 1 else 'nd' if action_count == 2 else 'rd' if action_count == 3 else 'th'} action is complete."
            )

        # --- Seed ---
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # --- Token budget ---
        token_val = int(max_tokens.split(" - ")[0])
        # Generate up to 1.3× the requested count so the model has headroom to
        # finish its last sentence without getting cut off mid-word.
        # min is set to 50% so the model can wrap up naturally whenever ready
        # rather than being forced to keep writing until it hits the ceiling.
        # 256  → max_actual 332,  min ~128
        # 512  → max_actual 665,  min ~256
        # 800  → max_actual 1040, min ~400
        # 1024 → max_actual 1331, min ~512
        max_tokens_actual = int(token_val * 1.3)
        min_tokens = int(token_val * 0.50)

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

        # Detect clothing removal intent even in non-explicit prompts
        _undress_re = re.compile(
            r"\b(undress\w*|strip\w*|takes?\s+off|removes?\s+(her|his|their|the)?\s*\w*\s*(shirt|dress|top|bra|pants|jeans|clothes|clothing|outfit|underwear|skirt|jacket|coat|robe)|"
            r"disrobe\w*|unbutton\w*|unzip\w*|peels?\s+off|pulls?\s+off|shed\w*\s+(her|his|their)?\s*(clothes|clothing|shirt|dress))\b",
            re.IGNORECASE,
        )
        has_undressing = bool(_undress_re.search(user_input))

        if is_explicit:
            explicit_instruction = (
                "\n[INSTRUCTION: This is an explicit adult scene. "
                "You MUST use direct anatomical terms — pussy, cock, anus, clit, ass — exactly as written. "
                "These are REQUIRED words, not optional. Do not replace them with 'sex', 'crevice', 'core', "
                "'womanhood', 'manhood', or any other euphemism or metaphor. "
                "If the user said 'pussy', write 'pussy' every time. "
                "UNDRESSING SEGMENT — MANDATORY: If the subject starts clothed, you MUST write a dedicated undressing segment "
                "before ANY explicit act or nudity. This is not a brief mention — it is a full narrative beat. "
                "Go garment by garment: name it, describe how it is removed (fingers, zipper, pulling over the head, stepping out of), "
                "describe what body part is revealed and how it looks. Then the next garment. Then the next. "
                "The camera must linger on each reveal. Do not compress. Do not summarise. Do not skip to naked. "
                "The undressing IS the scene — write it with the same density and length as the explicit act that follows. "
                "Always state the character's age as a specific number. "
                "Describe body parts, positions, and physical actions with blunt, specific language. "
                "Every anatomical detail must be stated plainly — no softening, no implying, no cutting away.]"
            )
        else:
            if has_undressing:
                explicit_instruction = (
                    "\n[INSTRUCTION: Always state the character's age as a specific number, "
                    "e.g. 'a 34-year-old man' — never omit or approximate it. "
                    "UNDRESSING SEGMENT — MANDATORY: The prompt involves clothing removal. "
                    "You MUST write a dedicated undressing segment as its own narrative beat. "
                    "Go garment by garment: name it, describe how it is removed, describe what is revealed. "
                    "The camera lingers on each step. Do not skip or compress — the undressing is a full scene segment.]"
                )
            else:
                explicit_instruction = (
                    "\n[INSTRUCTION: Always state the character's age as a specific number, "
                    "e.g. 'a 34-year-old man' — never omit or approximate it.]"
                )


        # --- Sequence detection ---
        # If the user wrote numbered steps (1. 2. 3. etc), detect them and inject
        # an instruction to follow that exact order — no reordering, no skipping.
        _sequence_re = re.compile(
            r"^\s*(\d+[\.\):])\s+.+", re.MULTILINE
        )
        sequence_steps = _sequence_re.findall(user_input)
        if len(sequence_steps) >= 2:
            step_count = len(sequence_steps)
            sequence_instruction = (
                f"\n[SEQUENCE INSTRUCTION: The user has provided {step_count} numbered steps. "
                f"You MUST follow them in exact order — step 1 first, then step 2, and so on. "
                f"Do not reorder, skip, or merge steps. Each step is one distinct beat in the scene. "
                f"Do not add actions before step 1 or after step {step_count}.]"
            )
        else:
            sequence_instruction = ""

        # --- Multi-subject detection ---
        # If the input describes two or more people, inject a spatial instruction
        # so the model tracks who is doing what and where they are relative to
        # each other and the camera — otherwise it tends to lose track.
        _multi_re = re.compile(
            r"\b(two\s+(women|men|people|girls|guys|characters|figures)|"
            r"both\s+(of\s+them|women|men|girls|guys)|"
            r"(she|he)\s+and\s+(she|he|her|him)|"
            r"(a\s+man\s+and\s+a\s+woman|a\s+woman\s+and\s+a\s+man)|"
            r"(a\s+man\s+and\s+a\s+man|a\s+woman\s+and\s+a\s+woman)|"
            r"couple|trio|they\s+(kiss|touch|embrace|undress|fuck|have))\b",
            re.IGNORECASE,
        )
        has_multi_subject = bool(_multi_re.search(user_input + " " + scene_context))
        if has_multi_subject:
            multi_instruction = (
                "\n[MULTI-SUBJECT INSTRUCTION: This scene has two or more people. "
                "For EACH person establish: their position in the frame (left/right/foreground/background), "
                "their spatial relationship to the other person (facing, beside, behind, above, etc.), "
                "and keep track of who is doing what throughout — never let actions become ambiguous. "
                "When referring back to them use consistent descriptors (e.g. 'the dark-haired woman', "
                "'the taller man') — not just 'she' or 'he' which causes confusion with two subjects.]"
            )
        else:
            multi_instruction = ""

        # --- Dialogue instruction ---
        if invent_dialogue:
            dialogue_instruction = (
                "\n\n[DIALOGUE INSTRUCTION: Invent dialogue that fits this scene naturally. "
                "Write it as inline prose woven into the action — NOT as a [DIALOGUE: ...] tag or bracketed block. "
                "The spoken words sit inside the sentence with attribution and physical delivery, like a novel. "
                "Examples: "
                "'He leans back, satisfied, \"I think I\\'ll have to go back tomorrow for more,\" he chuckles, his eyes crinkling at the corners.' "
                "'\"Don\\'t stop,\" she breathes, gripping the sheets, her voice barely above a whisper.' "
                "If the scene is sexual or explicit, dialogue must reflect that — breathless, reactive, commanding. "
                "Never write a bare floating quote. Never use [DIALOGUE: ...] tags. Dialogue is part of the prose, always.]"
            )
        else:
            has_user_dialogue = bool(re.search(r'["\u201c\u201d]', user_input))
            if has_user_dialogue:
                dialogue_instruction = (
                    "\n\n[DIALOGUE INSTRUCTION: Use ONLY the dialogue the user provided — do not invent or add any additional spoken words. "
                    "Place their exact words naturally in the scene as inline prose with attribution and delivery. "
                    "Examples: 'She smiles, \"I\\'m so happy,\" her voice bright, eyes wide.' "
                    "'\"I\\'m so happy,\" he whispers, pulling her close, his voice low.' "
                    "Never use [DIALOGUE: ...] tags. Weave the words into the action as part of the prose.]"
                )
            else:
                dialogue_instruction = (
                    "\n\n[DIALOGUE INSTRUCTION: No dialogue in this scene. No spoken words. "
                    "Describe only ambient sound — maximum 2 sounds. Format: [AMBIENT: sound 1, sound 2]]"
                )

        # Tell the model the token budget AND the hard action cap together
        # so both constraints are visible in the same instruction block.
        length_instruction = (
            f"\n[PACING — THIS IS MANDATORY: {pacing_hint} "
            f"Write approximately {token_val} tokens total. "
            f"Do not exceed the action count above under any circumstances.]"
        )

        # --- Merge vision context if provided ---
        # When a scene_context is wired in from the Vision Describe node,
        # prepend it so the LLM uses it as the authoritative subject/scene
        # description rather than inventing one from scratch.
        if scene_context and scene_context.strip():
            effective_input = (
                f"[SCENE CONTEXT FROM IMAGE — use this as the authoritative description "
                f"of the subject and setting; do not invent or contradict it]\n"
                f"{scene_context.strip()}\n\n"
                f"[USER DIRECTION — apply this as action, style, and mood over the above scene]\n"
                f"{user_input.strip()}"
            )
        else:
            effective_input = user_input.strip()

        # --- LoRA trigger injection ---
        # If the user provided trigger words, inject them as a hard instruction
        # so they appear at the start of the final prompt and are never buried.
        if lora_triggers and lora_triggers.strip():
            lora_instruction = (
                f"\n[LORA INSTRUCTION: You MUST begin the prompt output with these exact trigger words "
                f"before anything else: {lora_triggers.strip()} — place them as the very first words of your output, "
                f"then continue with the scene description immediately after.]"
            )
        else:
            lora_instruction = ""

        # --- Build messages ---
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": effective_input + sequence_instruction + multi_instruction + dialogue_instruction + explicit_instruction + lora_instruction + length_instruction},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        input_length = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                min_new_tokens=min_tokens,
                max_new_tokens=max_tokens_actual,
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

        # --- Build negative prompt ---
        neg_prompt = _build_negative_prompt(result, user_input)

        if not keep_model_loaded:
            self.unload_model()

        return (result, result, neg_prompt)


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
