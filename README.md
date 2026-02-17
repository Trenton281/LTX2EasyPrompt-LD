# LTX2EasyPrompt-LD



<img width="359" height="353" alt="image" src="https://github.com/user-attachments/assets/f19e6495-7a5c-479a-952b-6d13e9b42ed2" />



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
