# LoRA-Dataset-Forge

> **Two photos in. A trainer-ready LoRA dataset out.**
> Your character, rendered a thousand ways — with the same face every time.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)
![Target](https://img.shields.io/badge/target-Qwen--Image%20LoRA-ff7a59.svg)

---

## The problem it solves

Training a character LoRA takes 30 to 100 consistent images of one person. Getting those images is the hard part.

A photoshoot is expensive and inconsistent. Stable-diffusion-scraped photos drift between generations. Hand-captioning is tedious. Bucketing is an afterthought. By the time you've wrangled all of it, you've spent two days on dataset prep and you haven't even started training.

**LoRA-Dataset-Forge compresses that into an afternoon.** Drop a face shot and a body shot of each character into a folder, hit Generate, come back to a clean, captioned, aspect-bucketed, identity-locked training corpus that drops straight into musubi-tuner, ai-toolkit, or kohya.

---

## What you get

- **Dual-reference identity lock** — face + body references sent on every call, so the same person appears in every image
- **4+ billion unique specs** — framing × angle × expression × pose × lighting × environment × weather × prop × outfit, deterministic per character
- **Smart aspect ratios** — 3:4 close-ups, 4:5 mid-shots, 2:3 three-quarter body, 9:16 full-body. Qwen-Image's native buckets, no letterboxing
- **Qwen-native captions** — single-line natural language, trigger word first, auto-written to `.txt` sidecars. Optional vision-model re-captioning so captions match what actually got generated (not just what we asked for)
- **Review grid with face validation** — thumbnail wall, click to reject, Gemini-judge scores each image against your face reference, auto-flag low scorers
- **Per-character distribution control** — split 30 images into "10 close-ups + 10 full-body + 10 random" with one click; the forge remembers and refills rejected slots with guaranteed-unused specs
- **Parallel workers** — 4-way default, capped at 8. 700 images in ~15 minutes instead of an hour
- **Live cost estimator** — knows what it'll cost before you hit Generate
- **Auto-save** — every setting persists across sessions. API key opt-in
- **One-click trainer config export** — `_musubi_config.toml`, `_aitoolkit_config.yaml`, and a training README with the exact launch command
- **Dark UI** that isn't an eyesore, on Windows 11 with a dark titlebar to match

---

## The pipeline

```
    face.jpg      body.jpg         10 framings × 10 angles × 14 expressions
        \           /                × 18 lightings × 52 scenes × 36 props
         \         /                 × 10 weathers × 32 outfits
          \       /                      (~4 billion specs, deterministic
           v     v                        per character trigger word)
        ┌───────────┐                            │
        │           │<───────────────────────────┘
        │   forge   │
        │           │ system prompt: identity lock,
        └─────┬─────┘ anatomy rules, anti-AI-gloss
              │
              │ parallel worker pool (x4)
              ▼
     ┌──────────────────┐
     │  Nano Banana 2   │  gemini-3.1-flash-image-preview
     │  (Gemini Image)  │  at 2K, smart aspect per framing
     └────────┬─────────┘
              │
              ▼                                 ┌───────────┐
     ┌──────────────────┐   optional hop to    │ gemini-2.5 │
     │  generated image │──────────────────────>│  -flash    │
     │                  │    (re-caption,       │ (vision    │
     └────────┬─────────┘     validate face)    │  judge)    │
              │                                 └─────┬─────┘
              ▼                                       │
     ┌──────────────────┐                             │
     │  NNN.png +       │<────────────────────────────┘
     │  NNN.txt         │
     │  (+ manifest.json│
     │   with category, │
     │   score, flat id)│
     └────────┬─────────┘
              │
              │ export train configs
              ▼
     ┌──────────────────┐
     │  musubi-tuner    │  or ai-toolkit, kohya, OneTrainer
     │  Qwen-Image LoRA │
     └──────────────────┘
```

---

## Install

### Windows

```cmd
install.bat
run.bat
```

That's it. `install.bat` creates a local `.venv` and installs the two dependencies; `run.bat` launches the app.

### macOS / Linux

```bash
chmod +x install.sh run.sh
./install.sh
./run.sh
```

### Manual (any OS)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt   # Windows: .venv\Scripts\pip
.venv/bin/python forge.py
```

Needs Python 3.10+. Dependencies are `google-genai` and `Pillow` — nothing else.

---

## Setup

**1. Get a Gemini API key** at <https://aistudio.google.com/apikey>. Free tier works; paid tier is faster and has higher rate limits.

**2. Either** export it as `GEMINI_API_KEY` **or** paste it into the `API KEY` field in the app. If you want it remembered across sessions, flip the `remember` toggle next to the field (off by default for safety).

**3. Organize your dataset root** as one subfolder per character, each containing a face shot and a body shot:

```
My LoRA Dataset/
├── Anna/
│   ├── anna_face.jpg      (close-up, just the face)
│   └── anna_body.jpg      (full body, well-lit)
├── Bob/
│   ├── bob_portrait.jpg
│   └── bob_fullbody.jpg
└── ...
```

The forge auto-detects which is face vs body by filename keywords (`face`, `portrait`, `closeup`, `head` → face; `body`, `full`, `pose` → body). If your files don't contain those keywords, use the `edit` button next to each picker in the GUI.

---

## Quickstart

```
 1. launch → app scans your dataset root, lists each character as a row
 2. pick your model (Nano Banana 2 = fast, Pro = higher fidelity at 3x cost)
 3. (optional) click 'dist' on any character to set a custom framing mix
 4. hit GENERATE — watch the cost estimator, progress bar, and log
 5. when it finishes, click 'review' on a character
 6. click 'validate' → Gemini scores each image's face-match
 7. click 'flag low scores' → auto-rejects anything below the threshold
 8. click 'apply & close' → deletes rejected files
 9. hit GENERATE again → refills the gaps with guaranteed-unused specs
10. click 'export train configs' → drops musubi-tuner + ai-toolkit configs
    into your output folder, with the exact training command
```

That's the whole pipeline. Five minutes of clicking, an afternoon of waiting, one clean LoRA-ready dataset per character.

---

## Feature deep-dive

### Dual-reference identity lock
Every API call sends **both** your face and body references, flanked by a system prompt that's tuned hard for identity preservation (bone structure, eye shape, hair, skin tone, body proportions). Nano Banana 2 was specifically trained for multi-image conditioning and handles this better than any single-reference approach. Identity drift is rare and mostly catchable in the review pass.

### The combinatorial engine
Instead of recycling a small pool of prompts, the forge samples from the Cartesian product of 9 independent axes. That gives roughly 4 billion unique specs — at any count up to 500 (or higher), every image has a unique spec. Sampling is deterministic per character trigger word, so re-running with a larger count extends the prior sequence instead of reshuffling it.

### Smart aspect ratios
Close-ups at 3:4 keep face pixel density high. Mid-shots at 4:5 don't square-crop the shoulders. Full-body at 9:16 gives tall subjects actual vertical real estate. Qwen-Image buckets natively at all of these, so the trainer sees clean crops instead of letterboxed junk.

### Review + validate + regenerate
The review window is a clickable thumbnail wall. `validate` spins up a Gemini vision judge (`gemini-2.5-flash`) that rates each image's face similarity against your reference on a 0.00-1.00 scale, with one-sentence reasons logged into the manifest. `flag low scores` bulk-rejects anything below 0.60. `apply & close` deletes rejected files. Then the main Generate button fills the gaps with **guaranteed-unused** combinatorial specs — the manifest tracks every flat-index ever used, so regen never repeats a rejected spec.

### Per-character distribution
Click `dist` on any row. Four spinboxes: close, mid, full, random. Set `10·0·10·10` and you get slots 1-10 as close-ups, 11-20 as full-body, 21-30 as random. Category is recorded per slot in the manifest, so changing the distribution later doesn't silently reshuffle existing slots — refills still respect the original category.

### Trainer config export
Click `export train configs`. Drops three files into the output directory:

- `_musubi_config.toml` — ready-to-use for musubi-tuner's `--dataset_config`
- `_aitoolkit_config.yaml` — the `datasets:` block to merge into your training YAML
- `_training_README.md` — the full `accelerate launch` command with sensible defaults (LoRA rank 16, 10 epochs, bf16)

No guessing at bucketing settings, no hunting for the right arguments.

---

## Configuration

All settings auto-save to `_settings.json` next to the tool (gitignored — don't commit it).

| Setting | Default | What it controls |
|---|---|---|
| Model | `gemini-3.1-flash-image-preview` | Nano Banana 2. Switch to Pro for higher fidelity at ~3× cost. |
| Resolution | `2K` | Sweet spot for Qwen-Image's 1328px native resolution. |
| Aspect mode | `smart (per framing)` | 3:4 close / 4:5 mid / 2:3 3/4-body / 9:16 full-body. |
| Workers | `4` | Parallel API calls per character. Safe up to 8. |
| Verify captions | off | Re-caption each image via vision model. +~$0.0001 / image. |
| Remember API key | off | Off = memory-only for this session. On = saved to disk. |
| Default count | `30` | Applied to all characters via the "apply to all" button. |

---

## Cost

Rough per-image numbers for Nano Banana 2 at 2K:

| dataset | cost |
|---|---|
| 30 images (one character) | ~$1.20 |
| 210 images (7 characters × 30) | ~$8.20 |
| 700 images (7 × 100) | ~$27 |
| Same 700 with `verify captions` on | ~$27.10 |

Nano Banana Pro is roughly 3× more per image — worth it for a final polish pass, overkill for prototyping. See the live cost estimator in the app for the actual number on your current batch.

---

## Tips, gotchas, and honest advice

- **Don't put your dataset root inside OneDrive or Dropbox.** Sync locks will cause intermittent `PermissionError` during generation. The app shows an amber warning banner when it detects this. Use a local path like `C:/LoRA-Datasets/`.
- **Start with one character at count=5** to sanity-check your references before committing budget to the full batch. The forge's resume logic means a small test run isn't wasted — bump the count later and it extends the sequence.
- **Face validation is cheap.** ~$0.01 for a 100-image pass. Run it before rejecting manually — it catches identity drift you'd miss on a first visual scan.
- **Toggle `vary outfit` on** if you want inference-time wardrobe flexibility — the LoRA will learn the character decoupled from clothes, letting you prompt "ck_alice in a kimono" afterward. Leave it **off** if you want the reference outfit baked into the character identity (simpler prompting, less flexibility).
- **Watch the cost estimator** when switching models. Pro at 4K for 700 images is real money; nothing about the UX prevents you from clicking yourself into a $60 invoice.
- **Refusals happen.** Some framings or backgrounds trigger Gemini safety filters. The forge detects and skips these cleanly without wasting retries — they show up as `REFUSED` lines in the log and get marked in the manifest. Hit Generate again to refill the gap with a different spec.

---

## Known limits

**Nano Banana house style bakes in.** Every generated image carries subtle Gemini 3.1 Flash aesthetic signatures (mild over-saturation, specific skin-smoothing, particular catchlight rendering). Your LoRA will learn those alongside the character. Mitigations: mix in 20-30% real photos if you have them, use the Pro model, add light noise/grain in post, or accept it as the cost of synthetic data.

**Face validation is a heuristic, not an oracle.** Gemini-as-judge is good enough to catch obvious identity drift but it's not InsightFace cosine similarity. Plan B (local InsightFace integration) is on the roadmap.

**Gemini pricing changes.** The cost estimator uses approximate rates. Check <https://ai.google.dev/pricing> for current numbers.

**Windows-first.** Developed and tested on Windows 11. macOS/Linux scripts are included but less battle-tested. File bug reports if something breaks.

---

## Contributing

Codebase is two files:

- **[forge.py](forge.py)** — GUI, engine, workers, manifest, review, distribution editor, settings, export
- **[prompts.py](prompts.py)** — prompt library, combinatorial sampling, framing categories, caption renderer

The easiest high-impact contribution is adding entries to the axis pools in `prompts.py`:

```python
FRAMINGS = [...]       # 10 entries → add a "detail shot of hands" and it ripples
SCENES = [...]         # 52 entries → add a ski slope or a subway car
OUTFITS = [...]        # 32 entries → add a wedding dress or a winter coat
PROPS = [...]          # 36 entries → add a skateboard or a vintage camera
LIGHTINGS = [...]      # 18 entries → add a neon sign or a fireplace glow
```

Each new entry multiplies variety for every character. Pull requests welcome.

If you want to go deeper, the [docs/audit.md](docs/audit.md) and [docs/fix-plan.md](docs/fix-plan.md) files document the architecture review from the last release pass — a good map of the codebase.

---

## License

[MIT](LICENSE). Use it, fork it, ship your own thing on top of it. The only ask is that you keep the license notice.

---

<sub>Built for characters who deserve consistent rendering.</sub>
