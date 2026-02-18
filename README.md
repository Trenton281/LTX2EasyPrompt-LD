# LTX2EasyPrompt-LD



<img width="244" height="326" alt="image" src="https://github.com/user-attachments/assets/43f5cda0-c13a-491f-a13b-586e1cc51c55" />


<img width="560" height="406" alt="image" src="https://github.com/user-attachments/assets/872cbf72-398d-4727-bb58-b12de93bbec2" />

New features in this release

IT CAN READ IMAGES TOO - For Image to video workflows

🎯 Negative prompt output pin
Automatic scene-aware negative prompt, no second LLM call. Detects indoor/outdoor, day/night, explicit content, shot type and adds the right negatives for each. Wire it straight to your negative encoder and forget about it.

🏷️ LoRA trigger word input
Paste your trigger words once. They get injected at the very start of every prompt, every single run. Never buried halfway through the text, never accidentally dropped.

💬 Dialogue toggle
On — the LLM invents natural spoken dialogue woven into the scene as inline prose with attribution and delivery cues, like a novel. Off — it uses only the quoted dialogue you provide, or generates silently. No more floating unattributed quotes ruining your audio sync.

⚡ Bypass / direct mode
Flip the toggle and your text goes straight to the positive encoder with zero LLM processing. Full manual control when you want it, one click to switch back. Zero VRAM cost in bypass mode.



How it works
Step 1 — Vision node analyses your starting frame
Drop in any image and the vision node (Qwen2.5-VL-3B, + Qwen2.5 7b runs fully locally) writes a scene context describing:

Visual style — photorealistic, anime, 3D animation, cartoon etc
Subject — age, gender, skin tone, hair, body type
Clothing, or nudity described directly if present
Exact pose and body position
What they're on or interacting with
Shot type — close-up, medium shot, wide shot etc
Camera angle — eye level, low angle, high angle
Lighting — indoor/outdoor, time of day, light quality
Background and setting

It unloads from VRAM immediately after so LTX-2 has its full budget back.
Step 2 — Prompt node uses that as ground truth
Wire the vision output into the Easy Prompt node and your scene context becomes the authoritative starting point. The LLM doesn't invent the subject or guess the lighting — it takes exactly what the vision node described and animates it forward from your direction.
You just tell it what should happen next:

"she slowly turns to face the camera and smiles"

And it writes a full cinematic prompt that matches your actual image — correct lighting, correct shot framing, correct subject — and flows naturally from there.







A ComfyUI custom node that turns plain English descriptions into fully structured, cinema-ready prompts for LTX-2 video generation — powered by a local, uncensored LLM with zero internet dependency after setup.

LTX-2 Easy Prompt — By LoRa-Daddy
A ComfyUI custom node that turns plain English into fully structured, cinema-ready LTX-2 prompts — powered by a local, uncensored LLM. No cloud. No subscriptions. No restrictions.

🎬 What It Does
Type a rough idea in plain English. Get back a fully detailed prompt with shot type, character description, scene atmosphere, camera movement, and generated audio/dialogue — all automatically paced to your exact frame count and wired straight into your LTX-2 pipeline.

✨ Key Features
🧠 Two Models Included

NeuralDaredevil 8B — maximum quality, richest detail, recommended for most users
Llama 3.2 3B — low VRAM, runs on almost anything, great for sharing with others
Switch between them from a dropdown — automatic VRAM unload/reload, no restart needed

⏱️ Smart Frame-Aware Pacing

Set your frame count once in the node
The FRAMES output pin passes the real number straight to your LTX-2 sampler
Prompt pacing adjusts automatically in the background — never manually sync again

🎙️ Always Generates Audio & Dialogue

Every prompt includes real ambient sound and invented dialogue that fits the scene
Say "no dialogue" in your input if you want silence — otherwise it writes it for you
Dialogue matches the mood: a whisper, a command, a confession, whatever fits

📡 Fully Offline After First Download

HuggingFace network calls are blocked at the module level — before Python even loads the library
Point it at your local snapshot folder and it never touches the internet again, even on ComfyUI startup
Works behind firewalls, no WinError 10013, no MaxRetryError

🔒 Clean Output Every Time

Uses hard token-ID stopping — the model is physically prevented from writing role delimiters like "assistant" into your output
Regex cleaner runs as a backup safety net
No more dirty prompts bleeding into your pipeline

🔥 No Content Restrictions

Both models use abliterated weights — safety filters removed at the model level, not just prompted around
Explicit scenes use direct language automatically — no euphemisms, no fading out
Clothed characters get a full undressing sequence before any explicit action
Age always stated as a specific number
Dialogue matches the energy of the scene

🎯 Built for LTX-2 Specifically

Prompt structure follows LTX-2's preferred order: style → camera → character → scene → action → movement → audio
Pacing is automatically adjusted so the prompt fills your clip correctly without over-writing


⚙️ Setup
1️⃣ Install
Clone or download this repo and drop the folder into your ComfyUI custom nodes directory:
ComfyUI/custom_nodes/LTX2EasyPrompt-LD/
├── LTX2EasyPromptLD.py
└── __init__.py
Or clone directly:
bashcd ComfyUI/custom_nodes
git clone https://github.com/seanhan19911990-source/LTX2EasyPrompt-LD
Restart ComfyUI. Find the node under: Add Node → LTX2 → LTX-2 Easy Prompt By LoRa-Daddy

2️⃣ First Run — Download Your Model

Set offline_mode → false
Pick your model from the dropdown
Hit generate — it auto-downloads from HuggingFace
Once downloaded, flip offline_mode back to true


3️⃣ ⚠️ IMPORTANT — Set Your Local Paths For Full Offline Mode
After your models have downloaded, you need to find their snapshot folders on your machine and paste the paths into the node. This is what allows fully offline operation with zero network calls.
At the bottom of the node you will see two path fields:
local_path_8b — paste the full path to your NeuralDaredevil 8B snapshot folder
local_path_3b — paste the full path to your Llama 3.2 3B snapshot folder
Your paths will look something like this — but with your own Windows username and your own hash folder name:
C:\Users\YOUR_USERNAME\.cache\huggingface\hub\models--mlabonne--NeuralDaredevil-8B-abliterated\snapshots\YOUR_HASH_FOLDER
C:\Users\YOUR_USERNAME\.cache\huggingface\hub\models--huihui-ai--Llama-3.2-3B-Instruct-abliterated\snapshots\YOUR_HASH_FOLDER
To find your exact paths:

Open File Explorer
Navigate to C:\Users\YOUR_USERNAME\.cache\huggingface\hub\
Open the model folder → open snapshots → copy the full path of the hash folder inside
Paste it into the matching field on the node


⚠️ If you leave these fields blank and have offline_mode ON, the node will try to load from HF cache which may still cause network errors on some setups. Always fill in both paths for guaranteed offline operation.


4️⃣ Wire It Up
PROMPT  ──→  LTX-2 text/prompt input
FRAMES  ──→  Set_frames node
PREVIEW ──→  Preview Text node (optional)

5️⃣ Generate
Type your idea in plain English. Set your frame count. Hit generate. That's it.
