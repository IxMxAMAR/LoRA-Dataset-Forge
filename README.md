# LoRA-Dataset-Forge

A Tkinter GUI that turns a face reference + body reference into a fully diverse, identity-locked character training dataset for LoRA fine-tuning — powered by Google's Nano Banana 2 (`gemini-3.1-flash-image-preview`) or Nano Banana Pro.

Built specifically for character LoRAs on Qwen-Image / Qwen-Image-Edit, but the output format works with any trainer that accepts a folder of images + matching `.txt` captions (kohya_ss, ai-toolkit, musubi-tuner, OneTrainer, etc.).

> _Screenshot goes here — add `docs/screenshot.png` and link it in when you have one._

## What it does

Point it at a folder of characters (one subfolder per character, each containing a face shot + a body shot) and it will, per character:

1. Generate N identity-locked training images by sending **both references** to Nano Banana 2 on every call, with a strong system prompt that locks facial identity and body proportions.
2. Vary framing, camera angle, expression, pose, environment, weather, lighting, and (optionally) outfit across a deterministic combinatorial shuffle that has **4+ billion unique combinations** — at count = 500, every image has a unique spec.
3. Write Qwen-Image-style single-line natural-language captions to `.txt` sidecars, with the trigger word first.
4. Pick a smart aspect ratio per framing (3:4 close-ups, 4:5 mid-shots, 2:3 three-quarter body, 9:16 full body) so your dataset lands in Qwen-Image's native buckets.
5. Let you review the output in a thumbnail grid, validate face similarity with a Gemini vision-model judge, click to reject bad ones, and refill the gaps with guaranteed-fresh specs (no repeats).
6. Export ready-to-use dataset configs for `musubi-tuner` (TOML) and `ai-toolkit` (YAML), plus a training README with launch commands.

## Feature list

- **Multi-character batch processing** — auto-scans a dataset root for subfolders, detects face/body images by filename keywords
- **Parallel generation** — configurable worker pool (default 4, cap 8) for 4× wall-time speedup
- **Refusal detection** — distinguishes Gemini safety refusals from transient errors; refusals skip cleanly instead of burning retries
- **Per-character distribution control** — explicit bucket split (e.g. 10 close-ups + 10 full-body + 10 random for a 30-image set)
- **Resumable** — manifest-driven state; interrupted batches pick up where they left off
- **Auto-save** — every setting change persists to `_settings.json` (debounced 400 ms). API key save is opt-in via a `remember` toggle
- **Vision-model caption verification (optional)** — re-captions each generated image based on what's actually visible (fixes prompt-vs-output drift)
- **Face similarity validation (optional)** — Gemini-as-judge scores each generated image against the face reference; auto-flag low scorers in the review grid
- **Cost estimator** — live dollar estimate before you hit Generate
- **Prompt preview** — eyeball the system prompt, spec, and rendered prompt/caption for any character before committing
- **Trainer config export** — one click produces configs for musubi-tuner and ai-toolkit
- **Dark UI** with Windows 11 native dark titlebar

## Requirements

- Python 3.10+
- A Google Gemini API key ([get one](https://aistudio.google.com/apikey))
- ~2-5 GB of disk per character for 2K output (at the default 30 images/char)

## Install

### Windows

```cmd
install.bat
```

This creates a local `.venv` and installs dependencies. Then:

```cmd
run.bat
```

### macOS / Linux

```bash
chmod +x install.sh run.sh
./install.sh
./run.sh
```

### Manual (any OS)

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt    # or .venv\Scripts\pip on Windows
.venv/bin/python forge.py
```

## Setup

1. **Get an API key** at <https://aistudio.google.com/apikey> (free tier available, paid gets faster + higher rate limits)
2. **Set `GEMINI_API_KEY`** as an environment variable, or paste it into the app's API KEY field
3. **Prepare your dataset root** — one subfolder per character:

   ```
   My Characters/
     Alice/
       alice_face.jpg      ← face closeup
       alice_body.jpg      ← full-body shot
     Bob/
       bob_portrait.jpg
       bob_fullbody.jpg
     ...
   ```

   Face/body detection looks for keywords in filenames (`face`, `portrait`, `closeup` → face; `body`, `full`, `pose` → body). If your filenames don't contain those, use the `edit` button next to each image slot in the GUI.

## Using the tool

1. Launch → the app auto-scans your dataset root and lists each character as a row.
2. (Optional) Click `dist` on a character to set a custom distribution like "10 close + 10 full + 10 random".
3. Set a resolution (2K recommended for Qwen-Image LoRA), aspect mode (leave on "smart"), and workers count.
4. Optionally enable **verify captions** if you want vision-model re-captioning.
5. Click **GENERATE**. Watch the log, per-character progress, and cost estimator.
6. When it's done, click **review** on any character to:
   - Inspect all generated images in a thumbnail grid
   - Click any thumbnail to reject it (red border)
   - Click **validate** to score each image's face similarity to the reference
   - Click **flag low scores** to auto-reject anything below 0.60
   - Click **apply & close** to delete rejected files
7. Hit **GENERATE** again to refill the rejected slots with fresh combinatorial specs (no repeats).
8. Click **export train configs** to generate `_musubi_config.toml`, `_aitoolkit_config.yaml`, and `_training_README.md`.

## Training the LoRA

The exported `_training_README.md` has full launch commands for both `musubi-tuner` (recommended for Qwen-Image) and `ai-toolkit`. Basic musubi flow:

```bash
accelerate launch --num_cpu_threads_per_process 1 qwen_image_train_network.py \
    --dataset_config "_musubi_config.toml" \
    --output_dir ./outputs \
    --output_name my_lora \
    --save_model_as safetensors \
    --network_module networks.lora_qwen_image \
    --network_dim 16 --network_alpha 16 \
    --learning_rate 1e-4 --max_train_epochs 10 \
    --mixed_precision bf16
```

## Configuration

All settings persist automatically to `_settings.json` (which is gitignored — do not commit it).

| Setting | Default | Notes |
|---|---|---|
| Model | `gemini-3.1-flash-image-preview` | Nano Banana 2. Switch to `gemini-3-pro-image-preview` for higher fidelity at 4× cost. |
| Resolution | `2K` | Sweet spot for Qwen-Image's 1328px native res. |
| Aspect mode | `smart (per framing)` | 3:4 close / 4:5 mid / 2:3 3/4-body / 9:16 full-body / 2:3 full-wide. |
| Workers | `4` | Parallel API calls per character. Safe up to 8 on paid Gemini tiers. |
| Verify captions | off | Enable to re-caption each image via a vision model. Adds ~1s per image, ~$0.0001 per call. |
| Remember API key | off | When off, the key is kept in memory only. Turn on only if you're the sole user of your machine. |

## Cost

Very rough per-image USD (Nano Banana 2 at 2K): **~$0.039**. A typical 30-image character set is ~$1.20. A 7-character × 30-image production run lands around **$8-10**.

Pro model is roughly 3× more. See the live cost estimator in the app.

## Security notes

- `_settings.json` contains your API key (when "remember" is enabled) and local paths. **Do not commit it.** The bundled `.gitignore` excludes it.
- The tool makes outbound calls only to Gemini's API. No telemetry, no third-party analytics.
- Face validation and caption verification reuse the same API key.

## Tips

- **Start with one character at low count** (5-10 images) to sanity-check your references and prompt before burning budget on the full batch.
- **If your dataset root is inside OneDrive/Dropbox**, move it to a local folder first — sync locks can cause intermittent `PermissionError` during generation. The app shows an amber warning banner when it detects OneDrive in the path.
- **Face validation cost is negligible** (~$0.01 for a 100-image pass) but adds ~30s wall time. Run it before rejecting manually — the auto-flag often catches identity drift you'd have missed.
- **If you're training a character LoRA that needs wardrobe flexibility at inference time**, enable `vary outfit` per character (cycles through 32 outfits). If you want the LoRA to bake in the reference outfit, leave it off.
- **For characters with a specific body type** (tall, plus-size, petite), the body reference does most of the work — make sure it's a clean, well-framed full-body shot.

## Known limitations

- **Nano Banana house style:** every generated image carries subtle Gemini 3.1 Flash aesthetic artifacts (mild over-saturation, specific skin-smoothing). Your LoRA will learn those alongside the character. To mitigate, mix 20-30% real photos if you have them, use the Pro model, or add light noise/grain in post.
- **Gemini pricing changes:** the cost estimator uses approximate rates. Check <https://ai.google.dev/pricing> for current numbers.
- **OneDrive compatibility:** generation works but sync can lock files at inopportune moments. Local paths are strongly recommended.
- **Windows-first:** developed and tested on Windows 11. macOS/Linux scripts included but less battle-tested — report issues.

## Contributing

Pull requests welcome. The codebase is two files: [`forge.py`](forge.py) (GUI + engine + workflow) and [`prompts.py`](prompts.py) (prompt library and sampling logic). Adding more scenes, outfits, props, or lighting setups to the pools in `prompts.py` is the easiest way to improve output diversity.

## License

[MIT](LICENSE)
