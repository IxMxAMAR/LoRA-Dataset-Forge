"""Portrait Pack — generate 10 curated face/close-up shots per character.

Adds to an existing dataset without touching combinatorial specs. Each
character gets 10 standardized head-and-shoulders portraits with a fixed
expression/angle matrix, consistent outfit + plain studio backdrop, and a
young-model-tuned system prompt.

Usage:
    python portrait_pack.py                 # all enabled characters
    python portrait_pack.py Anna Caro       # specific characters only
    python portrait_pack.py --workers 4     # override parallelism
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import sys
import time
from io import BytesIO
from pathlib import Path

# Use the tool's existing engine + helpers
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import forge as F  # noqa: E402
from PIL import Image  # noqa: E402


# -----------------------------------------------------------------------------
# The 10 curated portrait specs — ordered from straight-on to off-axis
# -----------------------------------------------------------------------------
PORTRAIT_SPECS: list[dict] = [
    {"angle": "front-facing, looking directly at the camera with eyes on the lens",
     "expression": "soft natural closed-lip smile, eyes warm and present"},
    {"angle": "front-facing, looking directly at the camera",
     "expression": "genuine warm smile with a hint of teeth, eyes slightly crinkled"},
    {"angle": "head turned slightly to the left (about 15 degrees), eyes still on the camera",
     "expression": "soft natural smile, relaxed composure"},
    {"angle": "head turned slightly to the right (about 15 degrees), eyes still on the camera",
     "expression": "soft natural smile, relaxed composure"},
    {"angle": "three-quarter left turn (about 35 degrees), eyes on the camera",
     "expression": "calm confident fashion-editorial expression"},
    {"angle": "three-quarter right turn (about 35 degrees), eyes on the camera",
     "expression": "calm confident fashion-editorial expression"},
    {"angle": "front-facing with chin slightly raised, eyes on the camera",
     "expression": "confident and composed, jaw relaxed"},
    {"angle": "front-facing with chin slightly lowered, eyes on the camera",
     "expression": "soft thoughtful expression, warm and gentle"},
    {"angle": "front-facing, looking directly at the camera",
     "expression": "quiet genuine laugh caught mid-moment, teeth visible"},
    {"angle": "front-facing, eyes glancing slightly past the camera",
     "expression": "soft warm natural expression, lips gently closed"},
]


# The "young-model studio portrait" system prompt — replaces the standard
# SYSTEM_PROMPT for this script only.
PORTRAIT_SYSTEM_PROMPT = """You are generating a curated close-up portrait for a young fashion model's LoRA training dataset. The same person appears in every image.

IDENTITY LOCK (non-negotiable):
- Preserve EXACTLY the facial identity from Reference Image 1 (face reference): bone structure, eye shape and eye color, nose, lips, jawline, brow line.
- Preserve hair color, texture, length, hairline from references.
- The character is a young fashion model, approximately 18-20 years old. Render her youthful, healthy, and age-appropriate.

STUDIO PORTRAIT RULES (these are fixed across all 10 shots):
- Framing: close-up portrait, head and shoulders in frame, tight on the face.
- Backdrop: plain neutral studio backdrop — soft light gray, off-white, or softly blurred neutral. No environmental clutter, no props, no text.
- Lighting: soft, even studio softbox lighting that's flattering to young skin. Gentle key + fill, minimal harsh shadows.
- Outfit: keep the same clothing shown in the body reference image. Do NOT vary outfits across these images.
- Composition: the character fills most of the frame. Sharp focus on the eyes. Both eyes visible and catch-light lit.

SKIN RULES (critical — this is the difference between this portrait pack and the general dataset):
- Skin must look young, smooth, healthy, and naturally radiant.
- AVOID exaggerated skin detail that reads as older: no deep wrinkles, no pronounced lines around eyes or mouth, no rough texture, no visible enlarged pores, no age spots.
- Subtle natural skin variation is fine — a hint of natural texture — but err on the side of youthful, flattering rendering.
- Do NOT over-smooth into plastic or airbrushed territory. Aim for "professionally-lit young model photograph", not "heavily retouched".
- Clean eyes, clean eyebrows, natural lips. No heavy makeup unless already visible on the face reference.

QUALITY:
- Photorealistic, single person, one image.
- Sharp focus on the face and eyes.
- Correct anatomy: five fingers, correct limb count (though body is mostly cropped out in this framing).
- No text, logos, watermarks, borders, captions, split-frames, collages, or UI elements.
- Single person only, looking at the camera unless the angle says otherwise.
"""


PORTRAIT_CATEGORY = "portrait_pack"


def build_portrait_prompt(angle: str, expression: str, trigger: str) -> str:
    return (
        f"Framing: close-up portrait, head and shoulders only, tight on the face.\n"
        f"Camera angle: {angle}.\n"
        f"Expression: {expression}.\n"
        f"Pose: standing or seated upright, shoulders squared to the camera, relaxed.\n"
        f"Lighting: soft even studio softbox lighting, flattering to young skin.\n"
        f"Backdrop: plain neutral studio backdrop (soft light gray or off-white).\n"
        f"Outfit: keep the same clothing shown in the body reference image.\n\n"
        f"Generate a single photorealistic close-up portrait of the same young model "
        f"from the reference images. The face identity must be pixel-consistent "
        f"with the face reference. Skin must be smooth, youthful, and healthy — no "
        f"aging markers, no exaggerated texture. Sharp focus on the eyes."
    )


def build_portrait_caption(angle: str, expression: str, trigger: str) -> str:
    caption = (
        f"{trigger}, close-up portrait of {trigger}, {angle}, {expression}, "
        f"soft studio lighting, plain neutral backdrop. photorealistic portrait."
    )
    return " ".join(caption.split())


def _next_available_slot(out_char: Path) -> int:
    """Find the first unused NNN.png slot number in the character's folder."""
    existing = {int(p.stem) for p in out_char.glob("[0-9][0-9][0-9].png") if p.stem.isdigit()}
    n = 1
    while n in existing:
        n += 1
    return n


def _process_one(engine, face_bytes, body_bytes, out_char, slot, spec, trigger,
                 image_size, aspect, verify_captions):
    """Generate one portrait, save PNG + TXT, return (slot, ok_or_reason)."""
    prompt_text = build_portrait_prompt(spec["angle"], spec["expression"], trigger)
    caption = build_portrait_caption(spec["angle"], spec["expression"], trigger)

    try:
        img_bytes = engine.generate(
            face_bytes, body_bytes, prompt_text, PORTRAIT_SYSTEM_PROMPT,
            image_size=image_size, aspect_ratio=aspect,
        )
    except F.RefusalError as e:
        return slot, f"refused: {e}"
    except Exception as e:
        return slot, f"error: {e}"

    out_img = out_char / f"{slot:03d}.png"
    out_txt = out_char / f"{slot:03d}.txt"
    try:
        with Image.open(BytesIO(img_bytes)) as im:
            im.save(out_img, format="PNG")
        final_caption = caption
        if verify_captions:
            try:
                re_cap = engine.caption_image(img_bytes, trigger)
                if re_cap and len(re_cap) > 20:
                    final_caption = re_cap
            except Exception:
                pass
        out_txt.write_text(final_caption, encoding="utf-8")
    except Exception as e:
        return slot, f"save failed: {e}"

    return slot, "ok"


def run_for_character(char, root, output_dir, engine, image_size, aspect_mode,
                      workers, verify_captions):
    name = char.name
    out_char = output_dir / name
    out_char.mkdir(parents=True, exist_ok=True)
    manifest = F.load_manifest(out_char)
    trigger = char.trigger
    print(f"\n=== {name} (trigger={trigger}) ===")

    try:
        face_bytes = F._load_jpeg_bytes(char.face)
        body_bytes = F._load_jpeg_bytes(char.body)
    except Exception as e:
        print(f"  [err] failed to load refs: {e}")
        return

    start_slot = _next_available_slot(out_char)
    print(f"  next available slot: {start_slot:03d}, generating {len(PORTRAIT_SPECS)} portraits")

    # Determine aspect (3:4 for close-ups unless user overrides)
    aspect = "3:4" if aspect_mode.startswith("smart") else aspect_mode

    slots_and_specs = [
        (start_slot + i, spec) for i, spec in enumerate(PORTRAIT_SPECS)
    ]

    t0 = time.time()
    ok = 0
    errs = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _process_one, engine, face_bytes, body_bytes, out_char,
                slot, spec, trigger, image_size, aspect, verify_captions,
            ): (slot, spec)
            for slot, spec in slots_and_specs
        }
        for fut in concurrent.futures.as_completed(futures):
            slot, result = fut.result()
            if result == "ok":
                ok += 1
                # Record in manifest — no flat index since this is outside
                # the combinatorial space.
                F.record_slot(
                    manifest, slot, flat=-1, kept=True,
                    category=PORTRAIT_CATEGORY,
                    source="portrait_pack",
                )
                F.save_manifest(out_char, manifest)
                print(f"  {slot:03d} ok")
            else:
                errs += 1
                print(f"  {slot:03d} {result}")

    dt = time.time() - t0
    print(f"  summary: {ok}/{len(PORTRAIT_SPECS)} saved in {dt:.1f}s "
          f"({errs} error(s))")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("chars", nargs="*",
                        help="Specific character names (default: all enabled)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel workers per character batch (default 8)")
    parser.add_argument("--image-size", default=None,
                        help="Override image size (1K/2K/4K/AUTO). Default: from settings.")
    parser.add_argument("--aspect", default="3:4",
                        help="Aspect ratio. Default 3:4 (close-up).")
    parser.add_argument("--no-verify", action="store_true",
                        help="Disable caption verification (faster, still saves templated captions).")
    args = parser.parse_args()

    # Load settings from the tool's settings file
    settings = F.load_settings()
    api_key = settings.get("api_key") or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print("[error] no API key. Set GEMINI_API_KEY env var or enable 'remember' in the app.")
        sys.exit(1)
    model = settings.get("model", F.DEFAULT_MODEL)
    image_size = args.image_size or settings.get("image_size", F.DEFAULT_IMAGE_SIZE)
    aspect_mode = args.aspect
    verify_captions = not args.no_verify and bool(settings.get("verify_captions", False))

    root = Path(settings.get("dataset_root") or F.DEFAULT_ROOT)
    output_dir = root / F.DEFAULT_OUTPUT_SUBDIR
    if not output_dir.exists():
        print(f"[error] output dir not found: {output_dir}")
        sys.exit(1)

    # Scan characters and filter
    chars, warnings = F.scan_characters(root)
    for w in warnings:
        print(f"[scan] {w}")

    saved_chars = settings.get("characters") or {}
    # Filter: CLI-specified names, or all that are enabled in settings
    if args.chars:
        wanted = set(args.chars)
        chars = [c for c in chars if c.name in wanted]
    else:
        chars = [c for c in chars
                 if saved_chars.get(c.name, {}).get("enabled", True)]
    # Apply saved triggers so captions use user's real trigger words
    for c in chars:
        s = saved_chars.get(c.name) or {}
        if s.get("trigger"):
            c.trigger = s["trigger"]

    if not chars:
        print("[info] no characters to process.")
        return

    print(f"Characters to process: {[c.name for c in chars]}")
    print(f"Model: {model} | size: {image_size} | aspect: {aspect_mode} | "
          f"workers: {args.workers} | verify_captions: {verify_captions}")
    print(f"Output: {output_dir}")
    print(f"Will add {len(PORTRAIT_SPECS)} portraits per character "
          f"(= {len(PORTRAIT_SPECS) * len(chars)} total images)")

    # Rough cost preview
    per_image = F.PRICING_PER_IMAGE.get(model, {}).get(image_size, 0.04)
    total_imgs = len(PORTRAIT_SPECS) * len(chars)
    est = total_imgs * per_image + (total_imgs * F.CAPTION_COST_PER_CALL if verify_captions else 0)
    print(f"Estimated cost: ~${est:.2f}")
    print()

    try:
        engine = F.GeminiEngine(api_key=api_key, model=model)
    except Exception as e:
        print(f"[error] engine init failed: {e}")
        sys.exit(1)

    t_all = time.time()
    for c in chars:
        run_for_character(c, root, output_dir, engine,
                          image_size, aspect_mode, args.workers, verify_captions)
    print(f"\n=== DONE in {time.time() - t_all:.0f}s ===")


if __name__ == "__main__":
    main()
