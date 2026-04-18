"""LoRA-Dataset-Forge

Tkinter GUI that generates LoRA training datasets using Nano Banana 2.
Face reference + body reference + varied prompt pool -> diverse, identity-locked
training images with matching Qwen-Image-style caption files.
"""

from __future__ import annotations

import base64
import concurrent.futures
import copy
import ctypes
import json
import os
import queue
import re
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageTk

import prompts as P


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------

DEFAULT_ROOT = Path.home() / "Pictures" / "LoRA Datasets"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
FACE_KEYWORDS = ("face", "portrait", "closeup", "close-up", "head")
BODY_KEYWORDS = ("body", "full", "fullbody", "full-body", "pose")

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"  # Nano Banana 2
MODEL_CHOICES = [
    "gemini-3.1-flash-image-preview",   # Nano Banana 2 (fast)
    "gemini-3-pro-image-preview",       # Nano Banana Pro (higher fidelity)
    "gemini-2.5-flash-image",           # Nano Banana 1
]

DEFAULT_COUNT = 30
DEFAULT_OUTPUT_SUBDIR = "_dataset_output"
MAX_RETRIES = 4
DEFAULT_WORKERS = 4
MAX_WORKERS_LIMIT = 8

SETTINGS_FILENAME = "_settings.json"
SETTINGS_VERSION = 1
SETTINGS_DEBOUNCE_MS = 400

# ---- Generation mode ----
MODE_SYNC = "sync"
MODE_BATCH = "batch"
MODE_CHOICES = [MODE_SYNC, MODE_BATCH]
DEFAULT_MODE = MODE_SYNC
BATCH_COST_MULTIPLIER = 0.5     # Google Batch API: 50% of standard pricing
BATCH_STATE_FILENAME = "_batch.json"
BATCH_POLL_SECONDS = 30
BATCH_COMPLETED_STATES = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
}

MANIFEST_FILENAME = "_manifest.json"
MANIFEST_VERSION = 1
VALIDATE_MODEL = "gemini-2.5-flash"
VALIDATE_THRESHOLD = 0.60       # below this score → auto-flag in review UI
THUMB_SIZE = (200, 260)         # review-grid thumbnail cell
REVIEW_COLS = 5

# Rough per-image USD pricing for estimation. Gemini pricing changes — treat as indicative.
PRICING_PER_IMAGE = {
    "gemini-3.1-flash-image-preview": {"AUTO": 0.019, "1K": 0.019, "2K": 0.039, "4K": 0.090},
    "gemini-3-pro-image-preview":     {"AUTO": 0.060, "1K": 0.060, "2K": 0.120, "4K": 0.240},
    "gemini-2.5-flash-image":         {"AUTO": 0.010, "1K": 0.010, "2K": 0.019, "4K": 0.045},
}
VALIDATE_COST_PER_CALL = 0.00015
CAPTION_COST_PER_CALL = 0.00015


def estimate_cost(total_images: int, model: str, size: str,
                  verify_captions: bool = False, mode: str = MODE_SYNC) -> float:
    per_img = PRICING_PER_IMAGE.get(model, {}).get(size, 0.04)
    if mode == MODE_BATCH:
        per_img *= BATCH_COST_MULTIPLIER
    total = total_images * per_img
    if verify_captions:
        # caption verify stays sync regardless of image mode
        total += total_images * CAPTION_COST_PER_CALL
    return total

IMAGE_SIZE_CHOICES = ["AUTO", "1K", "2K", "4K"]
DEFAULT_IMAGE_SIZE = "2K"
ASPECT_CHOICES = ["smart (per framing)", "1:1", "3:4", "4:5", "2:3", "9:16", "4:3", "3:2", "16:9"]
DEFAULT_ASPECT = "smart (per framing)"


# -------------------------------------------------------------------------
# Theme
# -------------------------------------------------------------------------

COLOR = {
    "bg":        "#0d1014",
    "surface":   "#151921",
    "surface2":  "#1b202a",
    "row":       "#1c2129",
    "row_alt":   "#20252f",
    "border":    "#272d39",
    "divider":   "#1f242e",

    "text":      "#e6e8ef",
    "text_dim":  "#a2a8b8",
    "muted":     "#6e7689",

    "accent":    "#ff7a59",
    "accent_hi": "#ff9376",
    "accent_lo": "#c25a40",

    "ok":        "#7ee787",
    "warn":      "#ffb454",
    "err":       "#ff6b6b",
    "info":      "#7ac4ff",
}

FONT_UI = "Segoe UI"
FONT_MONO = "Cascadia Mono"

COL_WIDTHS = [48, 140, 240, 240, 150, 70, 120, 230, 110]  # matches header + rows
COL_STICKY = ["", "w", "ew", "ew", "w", "w", "w", "", "e"]
COL_PAD_X = 10


# -------------------------------------------------------------------------
# Gemini engine (unchanged)
# -------------------------------------------------------------------------

def _is_transient(err: Exception) -> bool:
    s = str(err)
    return any(m in s for m in ("429", "500", "502", "503", "504", "DEADLINE_EXCEEDED", "UNAVAILABLE"))


class RefusalError(Exception):
    """Gemini returned no image data (safety filter, content policy, or genuine failure).
    Distinct from transient errors because retrying the same prompt will fail the same way.
    """


class GeminiEngine:
    def __init__(self, api_key: str, model: str):
        from google import genai
        self._genai = genai
        self._types = __import__("google.genai.types", fromlist=["types"])
        self.client = genai.Client(api_key=api_key, http_options={"timeout": 300000})
        self.model = model

    def generate(self, face_bytes: bytes, body_bytes: bytes, prompt_text: str,
                 system_prompt: str, image_size: str = "AUTO", aspect_ratio: str = "AUTO") -> bytes:
        types = self._types
        parts = [
            types.Part.from_text(text="--- [Reference Image 1: FACE identity lock] ---"),
            types.Part.from_bytes(data=face_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text="--- [Reference Image 2: BODY identity lock] ---"),
            types.Part.from_bytes(data=body_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text=prompt_text),
        ]

        img_kwargs = {}
        if image_size and image_size != "AUTO":
            img_kwargs["image_size"] = image_size
        if aspect_ratio and aspect_ratio != "AUTO":
            img_kwargs["aspect_ratio"] = aspect_ratio

        config_kwargs = {
            "response_modalities": ["IMAGE"],
            "system_instruction": [types.Part.from_text(text=system_prompt)],
        }
        if img_kwargs:
            config_kwargs["image_config"] = types.ImageConfig(**img_kwargs)
        config = types.GenerateContentConfig(**config_kwargs)

        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=parts, config=config)

                # First pass: look for inline image bytes
                for cand in (resp.candidates or []):
                    content = getattr(cand, "content", None)
                    if not content:
                        continue
                    for part in (content.parts or []):
                        inline = getattr(part, "inline_data", None)
                        if inline and inline.data:
                            return inline.data

                # No image came back. Harvest any text (often a refusal reason) and
                # raise a RefusalError — retrying won't help, caller should skip.
                refusal_text = _extract_refusal_text(resp)
                raise RefusalError(refusal_text or "no image data returned (likely safety refusal)")
            except RefusalError:
                raise  # never retry a refusal
            except Exception as e:
                last_err = e
                if _is_transient(e) and attempt < MAX_RETRIES - 1:
                    delay = 3.0 * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise
        raise last_err  # type: ignore[misc]

    def caption_image(self, img_bytes: bytes, trigger: str,
                      text_model: str = VALIDATE_MODEL) -> str:
        """Generate a Qwen-Image-style caption based on what's actually visible
        in the generated image (not what we asked for). Fixes caption-lies-to-reality
        drift when Gemini deviates from the spec.
        """
        types = self._types
        prompt = (
            f"Write a LoRA training caption for this image.\n\n"
            f"Start the caption with the exact token '{trigger}' then describe what "
            f"is visibly present: framing (close-up / bust / waist-up / half-body / "
            f"three-quarter body / full body), camera angle, expression, pose, outfit "
            f"details, environment, lighting mood, any visible props or hand placement.\n\n"
            f"Rules:\n"
            f"- Single flowing sentence in natural language (for Qwen-Image training).\n"
            f"- Do not invent details that aren't visible.\n"
            f"- Do not mention names or make identity claims about the person.\n"
            f"- Under 65 words.\n"
            f"- No preamble, no 'This image shows…'. Output the caption only.\n\n"
            f"Example:\n"
            f"{trigger}, close-up portrait of {trigger}, three-quarter left angle, "
            f"soft natural smile, seated on a wooden chair, cream cable-knit sweater, "
            f"bright window light, neutral studio backdrop. photorealistic portrait."
        )
        parts = [
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            types.Part.from_text(text=prompt),
        ]
        resp = self.client.models.generate_content(model=text_model, contents=parts)
        text = ""
        for cand in (resp.candidates or []):
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in (content.parts or []):
                t = getattr(part, "text", None)
                if t:
                    text += t
        caption = text.strip().strip('"').strip("'").strip()
        # Ensure the caption starts with the trigger — some responses drop it
        if not caption.lower().startswith(trigger.lower()):
            caption = f"{trigger}, {caption}"
        return " ".join(caption.split())

    def validate_face(self, face_ref_bytes: bytes, img_bytes: bytes,
                      text_model: str = VALIDATE_MODEL) -> tuple[float, str]:
        """Score 0.0-1.0 how closely the face in `img` matches `face_ref`.

        Uses a Gemini text model as judge. 1.0 = unambiguously the same person,
        0.5 = possibly same, 0.0 = clearly different. Focus on bone structure.
        """
        types = self._types
        parts = [
            types.Part.from_text(text="REFERENCE FACE:"),
            types.Part.from_bytes(data=face_ref_bytes, mime_type="image/jpeg"),
            types.Part.from_text(text="GENERATED IMAGE:"),
            types.Part.from_bytes(data=img_bytes, mime_type="image/png"),
            types.Part.from_text(text=(
                "Compare the face in the GENERATED IMAGE to the REFERENCE FACE. "
                "Focus strictly on immutable facial features: bone structure, eye shape and "
                "spacing, nose shape, jawline, brow line, lip shape. Ignore hair, expression, "
                "makeup, lighting, framing, and outfit.\n\n"
                "Respond with exactly one line in this format:\n"
                "SCORE=<0.00 to 1.00> REASON=<short phrase>\n\n"
                "Scale: 1.00 = unambiguously the same person. "
                "0.75 = almost certainly same. 0.50 = possibly same. "
                "0.25 = likely different. 0.00 = clearly different person."
            )),
        ]
        resp = self.client.models.generate_content(model=text_model, contents=parts)
        text = ""
        for cand in (resp.candidates or []):
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in (content.parts or []):
                t = getattr(part, "text", None)
                if t:
                    text += t
        m = re.search(r"SCORE\s*=\s*([01](?:\.\d+)?)", text)
        score = float(m.group(1)) if m else 0.0
        m2 = re.search(r"REASON\s*=\s*(.+?)(?:\n|$)", text)
        reason = (m2.group(1).strip() if m2 else text.strip())[:140]
        return score, reason


def _settings_path() -> Path:
    """Settings file lives next to forge.py."""
    return Path(__file__).with_name(SETTINGS_FILENAME)


def load_settings() -> dict:
    path = _settings_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {}
        return data
    except Exception:
        return {}


def save_settings(data: dict):
    path = _settings_path()
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def load_manifest(out_char: Path) -> dict:
    path = out_char / MANIFEST_FILENAME
    if not path.exists():
        return {"version": MANIFEST_VERSION, "slots": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("version", MANIFEST_VERSION)
        data.setdefault("slots", {})
        return data
    except Exception:
        return {"version": MANIFEST_VERSION, "slots": {}}


def save_manifest(out_char: Path, manifest: dict):
    """Atomic manifest write: temp + replace. Cleans up tmp on failure."""
    out_char.mkdir(parents=True, exist_ok=True)
    path = out_char / MANIFEST_FILENAME
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def record_slot(manifest: dict, slot: int, flat: int, kept: bool = True, **extra):
    key = f"{slot:03d}"
    entry = manifest["slots"].get(key, {})
    entry.update({"flat": flat, "kept": kept, **extra})
    manifest["slots"][key] = entry


def used_flats_in_manifest(manifest: dict) -> set:
    out = set()
    for info in manifest.get("slots", {}).values():
        flat = info.get("flat")
        if isinstance(flat, int):
            out.add(flat)
    return out


def _extract_refusal_text(resp) -> str:
    """Pull text parts + finish reason from a response for diagnostic logging."""
    chunks = []
    try:
        for cand in (resp.candidates or []):
            fr = getattr(cand, "finish_reason", None)
            if fr:
                chunks.append(f"finish_reason={fr}")
            content = getattr(cand, "content", None)
            if not content:
                continue
            for part in (content.parts or []):
                txt = getattr(part, "text", None)
                if txt:
                    chunks.append(txt.strip())
        pf = getattr(resp, "prompt_feedback", None)
        if pf:
            bk = getattr(pf, "block_reason", None)
            if bk:
                chunks.append(f"block_reason={bk}")
    except Exception:
        pass
    joined = " | ".join(c for c in chunks if c)
    return joined[:300]


# -------------------------------------------------------------------------
# Character scanning
# -------------------------------------------------------------------------

@dataclass
class Character:
    name: str
    folder: Path
    all_images: list[Path]
    face: Path
    body: Path
    trigger: str
    count: int = DEFAULT_COUNT
    vary_outfit: bool = False
    enabled: bool = True
    # When set, overrides random sampling with a bucketed mix:
    # {"close": int, "mid": int, "full": int, "random": int}
    # Total count is derived from sum of values.
    distribution: Optional[dict] = None


def _pick(images: list[Path], keywords: tuple[str, ...]) -> Optional[Path]:
    for img in images:
        name = img.stem.lower()
        if any(kw in name for kw in keywords):
            return img
    return None


_UNSAFE_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
# Unicode bidi / invisible format chars — removed to prevent visual spoofing
_INVISIBLE_CHARS = re.compile(r'[\u202a-\u202e\u2066-\u2069\u200b-\u200f\ufeff]')
# Windows reserved device names (case-insensitive). Creating paths with these
# names can redirect writes to the device or cause cryptic errors.
_WIN_RESERVED = {
    "CON", "PRN", "AUX", "NUL",
    *{f"COM{i}" for i in range(1, 10)},
    *{f"LPT{i}" for i in range(1, 10)},
}


def _sanitize_char_name(raw: str) -> str:
    """Return a character name safe to use as a filesystem path component.

    Strips path separators, leading dots, illegal Windows chars, invisible
    Unicode controls (bidi/zero-width), the legacy ' retouch' suffix, and
    rejects Windows reserved device names. Returns empty string if nothing
    usable remains.
    """
    clean = raw.replace(" retouch", "").strip()
    clean = clean.strip(". ")                 # no leading/trailing dots → no ".."
    clean = _UNSAFE_NAME_CHARS.sub("", clean)
    clean = _INVISIBLE_CHARS.sub("", clean)
    # Reject Windows reserved names (case-insensitive match on the stem)
    if clean.upper().split(".")[0] in _WIN_RESERVED:
        return ""
    # Truncate to a reasonable filesystem-path-component limit
    if len(clean) > 120:
        clean = clean[:120].rstrip(". ")
    return clean


def scan_characters(root: Path) -> tuple[list[Character], list[str]]:
    """Scan `root` for character subfolders.

    Returns (characters, skipped_warnings). Inaccessible subfolders are skipped
    with a warning rather than crashing the scan. Character names are sanitized
    to prevent path traversal attacks, and case-insensitive duplicates are
    filtered (NTFS / macOS default FS both case-fold).
    """
    chars: list[Character] = []
    warnings: list[str] = []
    seen_casefolded: set[str] = set()
    if not root.exists():
        return chars, warnings
    try:
        root_resolved = root.resolve()
    except OSError as e:
        warnings.append(f"cannot resolve dataset root ({e})")
        return chars, warnings
    try:
        subdirs = sorted(root.iterdir())
    except (PermissionError, OSError) as e:
        warnings.append(f"cannot list dataset root ({e})")
        return chars, warnings
    for sub in subdirs:
        try:
            if not sub.is_dir():
                continue
            if sub.name.startswith(("_", ".")):
                continue
            if sub.name == "__MACOSX":
                continue
            # Path-traversal guard: the resolved path must stay under root
            try:
                sub_resolved = sub.resolve()
                sub_resolved.relative_to(root_resolved)
            except (ValueError, OSError):
                warnings.append(f"skipped {sub.name!r}: path escapes dataset root")
                continue
            imgs = sorted(p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        except (PermissionError, OSError) as e:
            warnings.append(f"skipped {sub.name!r}: {e}")
            continue
        if not imgs:
            continue

        clean = _sanitize_char_name(sub.name)
        if not clean:
            warnings.append(f"skipped {sub.name!r}: name has no safe characters")
            continue

        # Case-insensitive duplicate detection (NTFS / APFS default both case-fold)
        folded = clean.casefold()
        if folded in seen_casefolded:
            warnings.append(f"skipped {sub.name!r}: case-insensitive duplicate of an earlier character")
            continue
        seen_casefolded.add(folded)

        face = _pick(imgs, FACE_KEYWORDS) or imgs[0]
        body = _pick(imgs, BODY_KEYWORDS) or (imgs[1] if len(imgs) > 1 else imgs[0])
        if face == body:
            warnings.append(
                f"{clean}: face and body references are the same file "
                f"({face.name}) — add a distinct full-body shot for better results"
            )
        chars.append(Character(
            name=clean,
            folder=sub,
            all_images=imgs,
            face=face,
            body=body,
            trigger=f"ck_{clean.lower().replace(' ', '_')}",
        ))
    return chars, warnings


# -------------------------------------------------------------------------
# Worker (unchanged)
# -------------------------------------------------------------------------

@dataclass
class ProgressMsg:
    kind: str
    char: str = ""
    text: str = ""
    current: int = 0
    total: int = 0


# Reference images go to Google at full resolution. Two compression passes are
# applied (quality 92 + optimize + progressive) which trim ~15-25% of file size
# with no visually detectable loss. Batch mode additionally uses the Files API
# so each ref is uploaded ONCE per submission and referenced from every slot —
# that's how the 200MB JSONL inline-upload problem actually gets solved.
REF_JPEG_QUALITY = 92


def _slot_file_complete(out_char: Path, slot: int, min_png_bytes: int = 1024) -> bool:
    """Return True iff slot N's PNG and TXT are both present AND the PNG is
    large enough to be a real image (not a 0-byte stub from a crashed write).
    Tolerates OSError (TOCTOU races, transient permissions) by treating them
    as "not complete" rather than crashing the outer loop.
    """
    img_path = out_char / f"{slot:03d}.png"
    txt_path = out_char / f"{slot:03d}.txt"
    try:
        if not (img_path.exists() and txt_path.exists()):
            return False
        return img_path.stat().st_size > min_png_bytes
    except OSError:
        return False


def _load_jpeg_bytes(path: Path, quality: int = REF_JPEG_QUALITY) -> bytes:
    """Load an image and re-encode as JPEG at `quality` with optimize + progressive.
    No resizing — dimensions are preserved. Returns raw JPEG bytes.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        from io import BytesIO
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=quality, optimize=True, progressive=True)
        return buf.getvalue()


def _run_one(engine, char, slot, flat, spec, outfit, out_char,
             image_size, aspect_mode, verify_captions,
             face_bytes, body_bytes, stop_event):
    """Generate one slot. Returns (kind, info). kind ∈ {ok, skip_refused, error, stopped}.

    Bytes for face/body references are passed explicitly rather than read from
    engine instance state — makes concurrent safety obvious and prevents any
    accidental cross-character contamination.
    """
    if stop_event.is_set():
        return "stopped", {}

    out_img = out_char / f"{slot:03d}.png"
    out_txt = out_char / f"{slot:03d}.txt"

    prompt_text = P.build_prompt_text(spec, char.trigger, outfit)
    caption = P.build_caption(spec, char.trigger, outfit)
    aspect = P.smart_aspect(spec["framing"]) if aspect_mode.startswith("smart") else aspect_mode

    t0 = time.time()
    try:
        img_bytes = engine.generate(
            face_bytes, body_bytes, prompt_text, P.SYSTEM_PROMPT,
            image_size=image_size, aspect_ratio=aspect,
        )
    except RefusalError as e:
        return "skip_refused", {"slot": slot, "flat": flat, "total": char.count, "reason": str(e)[:200]}
    except Exception as e:
        return "error", {"slot": slot, "flat": flat, "total": char.count, "err": str(e)[:300]}

    try:
        from io import BytesIO
        with Image.open(BytesIO(img_bytes)) as im:
            im.save(out_img, format="PNG")
        out_txt.write_text(caption, encoding="utf-8")
    except Exception as e:
        return "error", {"slot": slot, "flat": flat, "total": char.count, "err": f"save: {e}"}

    caption_verified = False
    if verify_captions:
        try:
            new_caption = engine.caption_image(img_bytes, char.trigger)
            if new_caption and len(new_caption) > 20:
                out_txt.write_text(new_caption, encoding="utf-8")
                caption_verified = True
        except Exception:
            pass  # keep templated caption on failure

    return "ok", {
        "slot": slot, "flat": flat, "total": char.count,
        "dt": time.time() - t0,
        "framing": spec["framing"],
        "aspect": aspect,
        "caption_verified": caption_verified,
    }


def run_job(chars, root, output_dir, api_key, model, image_size, aspect_mode,
            workers, verify_captions, msgq, stop_event):
    try:
        engine = GeminiEngine(api_key=api_key, model=model)
    except Exception as e:
        msgq.put(ProgressMsg(kind="error", text=f"Failed to init Gemini client: {e}"))
        msgq.put(ProgressMsg(kind="done"))
        return

    enabled = [c for c in chars if c.enabled]
    total_jobs = sum(c.count for c in enabled)
    msgq.put(ProgressMsg(
        kind="log",
        text=f"Starting batch: {len(enabled)} character(s), {total_jobs} image(s), "
             f"resolution={image_size}, aspect={aspect_mode}, workers={workers}, "
             f"verify_captions={verify_captions}."
    ))

    for char in enabled:
        if stop_event.is_set():
            break
        msgq.put(ProgressMsg(kind="char_start", char=char.name, total=char.count))
        msgq.put(ProgressMsg(kind="log",
            text=f"[{char.name}] trigger='{char.trigger}', face='{char.face.name}', "
                 f"body='{char.body.name}', outfit={'vary' if char.vary_outfit else 'locked'}"))

        out_char = output_dir / char.name
        out_char.mkdir(parents=True, exist_ok=True)

        try:
            face_bytes = _load_jpeg_bytes(char.face)
            body_bytes = _load_jpeg_bytes(char.body)
        except Exception as e:
            msgq.put(ProgressMsg(kind="error", char=char.name,
                                 text=f"[{char.name}] failed to load refs: {e}"))
            continue

        manifest = load_manifest(out_char)
        manifest["trigger"] = char.trigger
        manifest["char"] = char.name
        manifest["vary_outfit"] = char.vary_outfit
        if char.distribution:
            manifest["distribution"] = dict(char.distribution)
        total_count = sum(char.distribution.values()) if char.distribution else char.count

        slots_to_fill = []
        for slot in range(1, total_count + 1):
            if not _slot_file_complete(out_char, slot):
                slots_to_fill.append(slot)

        used_flats = used_flats_in_manifest(manifest)

        if char.distribution:
            # Group missing slots by category, sample per-category with exclude.
            # Prefer category from manifest (original distribution at first generation)
            # so that changing the distribution later doesn't reshuffle what category
            # a given slot belongs to — maintaining consistency across regens.
            slots_by_cat: dict[str, list[int]] = {}
            for slot in slots_to_fill:
                existing = manifest.get("slots", {}).get(f"{slot:03d}", {})
                cat = existing.get("category") or P.slot_to_category(slot, char.distribution)
                slots_by_cat.setdefault(cat, []).append(slot)

            slot_jobs: dict[int, tuple] = {}
            running_exclude = set(used_flats)
            for cat, slots_in_cat in slots_by_cat.items():
                cat_dist = {c: 0 for c in P.CATEGORY_ORDER}
                cat_dist[cat] = len(slots_in_cat)
                cat_jobs = P.plan_jobs_distributed(
                    cat_dist, vary_outfit=char.vary_outfit,
                    trigger=char.trigger, exclude=running_exclude,
                )
                for slot, job in zip(slots_in_cat, cat_jobs):
                    slot_jobs[slot] = job  # (flat, spec, outfit, category)
                    running_exclude.add(job[0])
            # Assemble ordered list matching slots_to_fill order
            new_jobs_full = [(slot, *slot_jobs[slot]) for slot in slots_to_fill if slot in slot_jobs]
        else:
            plain_jobs = P.plan_jobs(len(slots_to_fill), vary_outfit=char.vary_outfit,
                                     trigger=char.trigger, exclude=used_flats)
            new_jobs_full = [(slot, flat, spec, outfit, None)
                             for slot, (flat, spec, outfit) in zip(slots_to_fill, plain_jobs)]

        if not slots_to_fill:
            msgq.put(ProgressMsg(kind="log",
                text=f"[{char.name}] all {total_count} slots already present; nothing to do."))
            msgq.put(ProgressMsg(kind="char_done", char=char.name))
            continue

        dist_note = ""
        if char.distribution:
            parts = [f"{cat}={char.distribution.get(cat, 0)}"
                     for cat in P.CATEGORY_ORDER if char.distribution.get(cat, 0)]
            dist_note = f", distribution[{', '.join(parts)}]"
        msgq.put(ProgressMsg(kind="log",
            text=f"[{char.name}] filling {len(slots_to_fill)} slot(s), "
                 f"excluding {len(used_flats)} previously used spec(s){dist_note}"))

        # Update total jobs count for accurate progress
        char_total_for_ui = len(new_jobs_full)

        done = 0
        refused = 0
        errors = 0
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers))
        try:
            futures = {}
            slot_category_map: dict[int, Optional[str]] = {}
            for slot, flat, spec, outfit, category in new_jobs_full:
                slot_category_map[slot] = category
                fut = pool.submit(_run_one, engine, char, slot, flat, spec, outfit,
                                  out_char, image_size, aspect_mode,
                                  verify_captions, face_bytes, body_bytes, stop_event)
                futures[fut] = slot

            for fut in concurrent.futures.as_completed(futures):
                if stop_event.is_set():
                    # cancel_futures=True skips any pending work; wait=False
                    # detaches from still-running futures so Stop completes
                    # immediately instead of blocking up to workers × 300s
                    # for Gemini calls to time out.
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    kind, info = fut.result()
                except Exception as e:
                    msgq.put(ProgressMsg(kind="error", char=char.name,
                                         text=f"[{char.name}] worker crash: {e}"))
                    errors += 1
                    continue

                if kind == "ok":
                    done += 1
                    cat = slot_category_map.get(info["slot"])
                    extra = {"category": cat} if cat else {}
                    record_slot(manifest, info["slot"], info["flat"], kept=True, **extra)
                    save_manifest(out_char, manifest)
                    msgq.put(ProgressMsg(kind="log",
                        text=f"[{char.name}] {info['slot']:03d} ok "
                             f"({info['dt']:.1f}s) · {info['framing']} / {info['aspect']}"
                             + (f" · [{cat}]" if cat else "")))
                    msgq.put(ProgressMsg(kind="progress", char=char.name,
                                         current=info["slot"], total=char_total_for_ui))
                elif kind == "skip_refused":
                    refused += 1
                    cat = slot_category_map.get(info["slot"])
                    extra = {"refused": True, "refuse_reason": info["reason"]}
                    if cat:
                        extra["category"] = cat
                    record_slot(manifest, info["slot"], info["flat"], kept=False, **extra)
                    save_manifest(out_char, manifest)
                    msgq.put(ProgressMsg(kind="log",
                        text=f"[{char.name}] {info['slot']:03d} REFUSED (not retrying): {info['reason']}"))
                    msgq.put(ProgressMsg(kind="progress", char=char.name,
                                         current=info["slot"], total=char_total_for_ui))
                elif kind == "error":
                    errors += 1
                    msgq.put(ProgressMsg(kind="error", char=char.name,
                        text=f"[{char.name}] {info['slot']:03d} failed: {info['err']}"))
        finally:
            # Always shut down the pool — explicit shutdown needed since we
            # dropped the `with` context manager above. wait=False is safe
            # here because the loop already drained all futures or was
            # explicitly cancelled via stop_event.
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass

        msgq.put(ProgressMsg(kind="char_done", char=char.name))
        msgq.put(ProgressMsg(kind="log",
            text=f"[{char.name}] summary — saved {done}, refused {refused}, errors {errors}"))

    msgq.put(ProgressMsg(kind="log", text="Batch complete."))
    msgq.put(ProgressMsg(kind="done"))


# -------------------------------------------------------------------------
# Batch mode (Google Batch API — 50% off, up to 24h turnaround)
# -------------------------------------------------------------------------

def _batch_state_path(output_dir: Path) -> Path:
    return output_dir / BATCH_STATE_FILENAME


def load_batch_state(output_dir: Path) -> Optional[dict]:
    p = _batch_state_path(output_dir)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_batch_state(output_dir: Path, state: dict):
    output_dir.mkdir(parents=True, exist_ok=True)
    p = _batch_state_path(output_dir)
    tmp = p.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, p)
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def clear_batch_state(output_dir: Path):
    p = _batch_state_path(output_dir)
    try:
        p.unlink(missing_ok=True)
    except Exception:
        pass


def _download_batch_result(engine: "GeminiEngine", result_file: str,
                           api_key: str, msgq) -> bytes:
    """Download a batch result file. Tries the SDK first; falls back to a
    direct HTTP GET on the download URI if the SDK download raises (known
    python-genai issue #1759 where batch result file names can exceed the
    Files API 40-char limit and the SDK rejects them).

    Returns the full file contents as bytes. The caller streams them out
    to a temp file immediately and frees the buffer, so peak RAM is the
    size of the result JSONL once.
    """
    try:
        return engine.client.files.download(file=result_file)
    except Exception as e:
        msg = str(e)
        msgq.put(ProgressMsg(kind="log",
            text=f"[batch] SDK download failed ({type(e).__name__}: {msg[:120]}), trying HTTP fallback..."))
    # Fallback: look up the file metadata (which still works for .get even if
    # .download doesn't), extract the download URI, fetch it via urllib
    # streaming in 1MB chunks so the HTTP response body doesn't also sit in
    # RAM alongside the eventual bytes object we return.
    try:
        f = engine.client.files.get(name=result_file)
    except Exception as e:
        raise RuntimeError(
            f"both SDK download and metadata lookup failed for {result_file}: {e}"
        )
    uri = getattr(f, "download_uri", None) or getattr(f, "uri", None)
    if not uri:
        raise RuntimeError(f"no download URI on batch result {result_file}")
    import urllib.request
    from io import BytesIO
    req = urllib.request.Request(uri, headers={"x-goog-api-key": api_key})
    buf = BytesIO()
    with urllib.request.urlopen(req, timeout=300) as resp:
        while True:
            chunk = resp.read(1 << 20)  # 1 MB at a time
            if not chunk:
                break
            buf.write(chunk)
    return buf.getvalue()


def _cleanup_ref_files(engine: "GeminiEngine", names: list[str]) -> int:
    """Delete uploaded Files API entries. Safe to call with any mix of names.
    Returns count successfully deleted. Silent on individual failures (Google
    auto-expires at 48h anyway).
    """
    if not names or engine is None:
        return 0
    deleted = 0
    for n in names:
        if not n:
            continue
        try:
            engine.client.files.delete(name=n)
            deleted += 1
        except Exception:
            pass
    return deleted


def _wait_for_file_active(engine: "GeminiEngine", file_name: str,
                          timeout_s: int = 90, msgq=None) -> None:
    """Poll the Files API until a just-uploaded file is ACTIVE (ready for use).
    Referencing a PROCESSING file in a batch request can cause per-request
    errors that still bill at batch rate — so we MUST wait before submitting.
    Raises RuntimeError on FAILED, TimeoutError past timeout_s.

    Surfaces unknown/unspecified states every ~10s so the user sees progress
    instead of a silent 90s stall.
    """
    start = time.time()
    last_notified_state: Optional[str] = None
    last_notify_time = start
    while time.time() - start < timeout_s:
        try:
            f = engine.client.files.get(name=file_name)
        except Exception:
            time.sleep(1.5)
            continue
        state = getattr(f, "state", None)
        state_name = getattr(state, "name", str(state)) if state else ""
        if state_name == "ACTIVE":
            return
        if state_name == "FAILED":
            err = getattr(f, "error", None)
            raise RuntimeError(f"Files API reported FAILED for {file_name}: {err}")
        # Surface unknown / long-waiting states to the user
        now = time.time()
        if msgq is not None and (state_name != last_notified_state or now - last_notify_time >= 10):
            try:
                msgq.put(ProgressMsg(kind="log",
                    text=f"[files] {file_name} state={state_name or 'UNSPECIFIED'} "
                         f"({now-start:.0f}s waited)"))
            except Exception:
                pass
            last_notified_state = state_name
            last_notify_time = now
        time.sleep(1.5)
    raise TimeoutError(f"file {file_name} never reached ACTIVE within {timeout_s}s")


def _upload_ref_via_files_api(engine: "GeminiEngine", jpeg_bytes: bytes,
                              display_name: str, msgq=None) -> tuple[str, str]:
    """Upload a ref to Google's Files API and return (uri, short_name).

    `uri` is the full URL used in fileData.fileUri for model requests.
    `short_name` is the 'files/abc123' form used for delete/get calls.
    Blocks until the file reaches ACTIVE state — otherwise the batch could
    reference a PROCESSING file and waste request budget on resolution errors.
    """
    types = engine._types
    # Files API requires a file path, not bytes — write to temp.
    # Nested try/finally guarantees the file handle is closed BEFORE we try
    # to unlink it: Windows refuses to unlink an open file and would silently
    # leak the temp file + handle until process exit if write() ever raised.
    tmp = tempfile.NamedTemporaryFile(
        prefix="ref_", suffix=".jpg", delete=False,
    )
    try:
        try:
            tmp.write(jpeg_bytes)
        finally:
            tmp.close()
        uploaded = engine.client.files.upload(
            file=tmp.name,
            config=types.UploadFileConfig(
                display_name=display_name,
                mime_type="image/jpeg",
            ),
        )
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    # Defensive: both .uri and .name populated on success.
    short_name = uploaded.name or ""
    uri = uploaded.uri or short_name
    if not short_name or not uri:
        raise RuntimeError(f"Files API upload returned empty identifier for {display_name}")

    # BLOCK until ACTIVE — referencing a PROCESSING file would bill for errors.
    _wait_for_file_active(engine, short_name, msgq=msgq)
    return uri, short_name


def _build_batch_request_uri(spec, trigger, outfit, face_uri: str, body_uri: str,
                             image_size: str, aspect: str) -> dict:
    """Build a batch request that references face+body via Files API URIs
    (as opposed to embedding base64 inline). Dramatically smaller payload
    when the same refs are reused across many requests.
    """
    prompt_text = P.build_prompt_text(spec, trigger, outfit)
    parts = [
        {"text": "--- [Reference Image 1: FACE identity lock] ---"},
        {"fileData": {"mimeType": "image/jpeg", "fileUri": face_uri}},
        {"text": "--- [Reference Image 2: BODY identity lock] ---"},
        {"fileData": {"mimeType": "image/jpeg", "fileUri": body_uri}},
        {"text": prompt_text},
    ]
    gen_config: dict = {"responseModalities": ["IMAGE"]}
    image_cfg: dict = {}
    if image_size and image_size != "AUTO":
        image_cfg["imageSize"] = image_size
    if aspect and aspect != "AUTO":
        image_cfg["aspectRatio"] = aspect
    if image_cfg:
        gen_config["imageConfig"] = image_cfg

    return {
        "contents": [{"parts": parts}],
        "systemInstruction": {"parts": [{"text": P.SYSTEM_PROMPT}]},
        "generationConfig": gen_config,
    }


def build_batch_jsonl(engine: "GeminiEngine", chars: list, output_dir: Path,
                      image_size: str, aspect_mode: str,
                      msgq: "queue.Queue[ProgressMsg]",
                      stop_event: threading.Event) -> tuple[Path, dict, list[str]]:
    """Construct the JSONL file and a key_map that maps each request key
    back to (char_name, slot, flat, category).

    Uses the Files API to upload each character's face+body refs ONCE per
    submission, then references them by URI from every slot request — so
    a 30-image batch uploads ~12 MB of refs + a tiny JSONL instead of
    30 copies of the same refs (200+ MB).

    Returns (jsonl_path, key_map, uploaded_file_names). Caller is responsible
    for cleanup.
    """
    key_map: dict[str, dict] = {}
    uploaded_names: list[str] = []
    tmp = tempfile.NamedTemporaryFile(
        prefix="forge_batch_", suffix=".jsonl", delete=False, mode="w", encoding="utf-8",
    )
    jsonl_path = Path(tmp.name)
    written = 0
    try:
        for char in chars:
            if stop_event.is_set():
                msgq.put(ProgressMsg(kind="log", text="stop requested during batch prep — aborting"))
                break
            out_char = output_dir / char.name
            out_char.mkdir(parents=True, exist_ok=True)
            manifest = load_manifest(out_char)
            manifest["trigger"] = char.trigger
            manifest["char"] = char.name
            manifest["vary_outfit"] = char.vary_outfit
            if char.distribution:
                manifest["distribution"] = dict(char.distribution)

            total_count = sum(char.distribution.values()) if char.distribution else char.count
            slots_to_fill = []
            for slot in range(1, total_count + 1):
                if not _slot_file_complete(out_char, slot):
                    slots_to_fill.append(slot)

            if not slots_to_fill:
                msgq.put(ProgressMsg(kind="log",
                    text=f"[{char.name}] all {total_count} slots present; skipping in batch"))
                continue

            try:
                face_bytes = _load_jpeg_bytes(char.face)
                body_bytes = _load_jpeg_bytes(char.body)
            except Exception as e:
                msgq.put(ProgressMsg(kind="error", char=char.name,
                                     text=f"[{char.name}] failed to load refs: {e}"))
                continue

            # Upload refs once per char via Files API → get URIs for reuse.
            # _upload_ref_via_files_api returns (full_uri, "files/xxx" short_name)
            # and blocks until each file reaches ACTIVE, so the batch never
            # references a still-processing file.
            msgq.put(ProgressMsg(kind="log",
                text=f"[{char.name}] uploading face ({len(face_bytes)/1024:.0f}KB) + "
                     f"body ({len(body_bytes)/1024:.0f}KB) to Files API..."))
            try:
                face_uri, face_name = _upload_ref_via_files_api(
                    engine, face_bytes, f"forge_{char.name}_face", msgq=msgq)
                body_uri, body_name = _upload_ref_via_files_api(
                    engine, body_bytes, f"forge_{char.name}_body", msgq=msgq)
                uploaded_names.extend([face_name, body_name])
                msgq.put(ProgressMsg(kind="log",
                    text=f"[{char.name}] refs ACTIVE on Google — proceeding"))
            except Exception as e:
                msgq.put(ProgressMsg(kind="error", char=char.name,
                                     text=f"[{char.name}] ref upload failed: {e}"))
                continue

            used_flats = used_flats_in_manifest(manifest)
            if char.distribution:
                slots_by_cat: dict[str, list[int]] = {}
                for slot in slots_to_fill:
                    existing = manifest.get("slots", {}).get(f"{slot:03d}", {})
                    cat = existing.get("category") or P.slot_to_category(slot, char.distribution)
                    slots_by_cat.setdefault(cat, []).append(slot)
                slot_jobs: dict[int, tuple] = {}
                running_exclude = set(used_flats)
                for cat, slots_in_cat in slots_by_cat.items():
                    cat_dist = {c: 0 for c in P.CATEGORY_ORDER}
                    cat_dist[cat] = len(slots_in_cat)
                    for slot, job in zip(slots_in_cat, P.plan_jobs_distributed(
                            cat_dist, vary_outfit=char.vary_outfit,
                            trigger=char.trigger, exclude=running_exclude)):
                        slot_jobs[slot] = job
                        running_exclude.add(job[0])
                jobs_full = [(slot, *slot_jobs[slot]) for slot in slots_to_fill if slot in slot_jobs]
            else:
                plain = P.plan_jobs(len(slots_to_fill), vary_outfit=char.vary_outfit,
                                    trigger=char.trigger, exclude=used_flats)
                jobs_full = [(slot, flat, spec, outfit, None)
                             for slot, (flat, spec, outfit) in zip(slots_to_fill, plain)]

            for slot, flat, spec, outfit, category in jobs_full:
                aspect = P.smart_aspect(spec["framing"]) if aspect_mode.startswith("smart") else aspect_mode
                key = f"{char.name}__{slot:03d}"
                request = _build_batch_request_uri(spec, char.trigger, outfit,
                                                   face_uri, body_uri, image_size, aspect)
                caption = P.build_caption(spec, char.trigger, outfit)
                line = json.dumps({"key": key, "request": request}, ensure_ascii=False)
                tmp.write(line + "\n")
                written += 1

                key_map[key] = {
                    "char": char.name,
                    "slot": slot,
                    "flat": flat,
                    "category": category,
                    "trigger": char.trigger,
                    "caption": caption,
                    "aspect": aspect,
                    "framing": spec["framing"],
                }

            # Save manifest prep (even pre-submit) so if app crashes we know chars
            save_manifest(out_char, manifest)
    finally:
        tmp.close()

    if written == 0:
        try:
            jsonl_path.unlink(missing_ok=True)
        except Exception:
            pass

    msgq.put(ProgressMsg(kind="log",
        text=f"Built batch JSONL with {written} request(s), {len(uploaded_names)} ref file(s) uploaded"))
    return jsonl_path, key_map, uploaded_names


def run_batch_submit(chars: list, output_dir: Path, api_key: str, model: str,
                     image_size: str, aspect_mode: str, verify_captions: bool,
                     msgq: "queue.Queue[ProgressMsg]", stop_event: threading.Event):
    """Build JSONL, upload, submit batch. Writes _batch.json state file.
    Emits kind='batch_submitted' or kind='error' then 'done'.
    """
    enabled = [c for c in chars if c.enabled]
    if not enabled:
        msgq.put(ProgressMsg(kind="error", text="no characters enabled for batch"))
        msgq.put(ProgressMsg(kind="done"))
        return

    try:
        engine = GeminiEngine(api_key=api_key, model=model)
    except Exception as e:
        msgq.put(ProgressMsg(kind="error", text=f"Failed to init Gemini client: {e}"))
        msgq.put(ProgressMsg(kind="done"))
        return

    msgq.put(ProgressMsg(kind="log",
        text=f"Preparing batch: {len(enabled)} character(s), "
             f"resolution={image_size}, aspect={aspect_mode}, model={model}"))

    ref_file_names: list[str] = []
    jsonl_path: Optional[Path] = None
    try:
        jsonl_path, key_map, ref_file_names = build_batch_jsonl(
            engine, enabled, output_dir, image_size, aspect_mode, msgq, stop_event)
    except Exception as e:
        msgq.put(ProgressMsg(kind="error", text=f"Failed to build batch JSONL: {e}"))
        # Clean up any refs that did upload before the failure
        deleted = _cleanup_ref_files(engine, ref_file_names)
        if deleted:
            msgq.put(ProgressMsg(kind="log", text=f"cleaned up {deleted} orphaned ref upload(s)"))
        msgq.put(ProgressMsg(kind="done"))
        return

    if not key_map:
        # No work to do — but refs may have been uploaded before the all-slots-present
        # check in build_batch_jsonl. Clean them up or they leak until Google auto-expires.
        deleted = _cleanup_ref_files(engine, ref_file_names)
        if deleted:
            msgq.put(ProgressMsg(kind="log", text=f"cleaned up {deleted} unused ref upload(s)"))
        if jsonl_path and jsonl_path.exists():
            try:
                jsonl_path.unlink(missing_ok=True)
            except Exception:
                pass
        msgq.put(ProgressMsg(kind="log", text="Nothing to batch — all slots already filled."))
        msgq.put(ProgressMsg(kind="done"))
        return

    if stop_event.is_set():
        msgq.put(ProgressMsg(kind="log", text="stop requested before JSONL upload — aborting"))
        _cleanup_ref_files(engine, ref_file_names)
        try:
            jsonl_path.unlink(missing_ok=True)
        except Exception:
            pass
        msgq.put(ProgressMsg(kind="done"))
        return

    jsonl_size = jsonl_path.stat().st_size if jsonl_path.exists() else 0
    msgq.put(ProgressMsg(kind="log",
        text=f"Batch JSONL ready: {jsonl_size/1_000_000:.1f} MB "
             f"(refs uploaded separately via Files API)"))

    # Upload JSONL
    uploaded = None
    try:
        msgq.put(ProgressMsg(kind="log", text=f"Uploading batch JSONL ({jsonl_path.stat().st_size:,} bytes)..."))
        types = engine._types
        uploaded = engine.client.files.upload(
            file=str(jsonl_path),
            config=types.UploadFileConfig(
                display_name=f"forge_batch_{int(time.time())}",
                mime_type="application/x-ndjson",
            ),
        )
    except Exception as e:
        msgq.put(ProgressMsg(kind="error", text=f"File upload failed: {e}"))
        _cleanup_ref_files(engine, ref_file_names)
        try:
            jsonl_path.unlink(missing_ok=True)
        except Exception:
            pass
        msgq.put(ProgressMsg(kind="done"))
        return
    finally:
        try:
            jsonl_path.unlink(missing_ok=True)
        except Exception:
            pass

    # Submit batch. Use the typed config class (not a bare dict) — the dict
    # form is coerced by the SDK but known to silently drop unknown fields
    # (python-genai issue #1389).
    try:
        types = engine._types
        batch_job = engine.client.batches.create(
            model=model,
            src=uploaded.name,
            config=types.CreateBatchJobConfig(
                display_name=f"lora-forge-{int(time.time())}",
            ),
        )
    except Exception as e:
        msgq.put(ProgressMsg(kind="error", text=f"Batch submit failed: {e}"))
        _cleanup_ref_files(engine, ref_file_names)
        # Also delete the uploaded JSONL file on Google's side since batch never consumed it
        try:
            engine.client.files.delete(name=uploaded.name)
        except Exception:
            pass
        msgq.put(ProgressMsg(kind="done"))
        return

    # Persist state
    state = {
        "version": 1,
        "batch_name": batch_job.name,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": model,
        "image_size": image_size,
        "aspect_mode": aspect_mode,
        "verify_captions": bool(verify_captions),
        "api_key_hint": f"len={len(api_key)}",  # do NOT persist the key itself
        "total_requests": len(key_map),
        "key_map": key_map,
        "ref_file_names": ref_file_names,  # Files API uploads for post-batch cleanup
    }
    try:
        save_batch_state(output_dir, state)
    except Exception as e:
        msgq.put(ProgressMsg(kind="log", text=f"WARN — state save failed ({e}); batch continues but cannot resume on close"))

    initial_state = getattr(getattr(batch_job, "state", None), "name", "UNKNOWN")
    msgq.put(ProgressMsg(kind="log",
        text=f"Batch submitted: {batch_job.name} · {len(key_map)} request(s) · "
             f"initial state={initial_state}"))
    # Deliberately NOT emitting 'done' here — the poller will emit it on
    # actual completion. Emitting 'done' now would show a misleading
    # "batch finished" log line while the batch is still processing.
    msgq.put(ProgressMsg(kind="batch_submitted",
                         text=batch_job.name, current=len(key_map)))


def run_batch_poll(output_dir: Path, api_key: str, model: str,
                   msgq: "queue.Queue[ProgressMsg]", stop_event: threading.Event):
    """Poll an already-submitted batch until completion or stop_event, then write results."""
    state = load_batch_state(output_dir)
    if not state:
        msgq.put(ProgressMsg(kind="error", text="no batch state found to poll"))
        msgq.put(ProgressMsg(kind="done"))
        return

    batch_name = state["batch_name"]
    key_map = state["key_map"]

    try:
        engine = GeminiEngine(api_key=api_key, model=model)
    except Exception as e:
        msgq.put(ProgressMsg(kind="error", text=f"Failed to init Gemini client for polling: {e}"))
        msgq.put(ProgressMsg(kind="done"))
        return

    msgq.put(ProgressMsg(kind="log", text=f"Polling batch {batch_name} every {BATCH_POLL_SECONDS}s..."))

    job = None
    start_time = time.time()
    poll_count = 0
    last_state: Optional[str] = None
    pending_note_shown = False

    while not stop_event.is_set():
        try:
            job = engine.client.batches.get(name=batch_name)
        except Exception as e:
            msgq.put(ProgressMsg(kind="log", text=f"poll transient error: {e}"))
            for _ in range(BATCH_POLL_SECONDS):
                if stop_event.is_set():
                    break
                time.sleep(1)
            continue

        poll_count += 1
        state_name = getattr(job.state, "name", str(job.state))
        elapsed = time.time() - start_time

        # Progress stats if Google reports them (not always populated for image jobs)
        stats = getattr(job, "completion_stats", None)
        stats_str = ""
        if stats is not None:
            done = getattr(stats, "successful_count", None) or getattr(stats, "succeeded_count", None) or 0
            failed = getattr(stats, "failed_count", None) or 0
            total = getattr(stats, "total_count", None) or getattr(stats, "total_requests", None)
            if total:
                stats_str = f" · completed {done}/{total} (failed {failed})"

        # Always compute staleness — makes it visible at every poll instead
        # of hiding it behind a 10-min threshold.
        stale_str = ""
        stale_sec = 0.0
        up = getattr(job, "update_time", None)
        if up is not None:
            import datetime as _dt
            now = _dt.datetime.now(_dt.timezone.utc)
            try:
                stale_sec = (now - up).total_seconds()
                if stale_sec >= 30:
                    stale_str = f" · server last updated {stale_sec/60:.1f}min ago"
            except Exception:
                pass

        msgq.put(ProgressMsg(kind="batch_status",
                             text=f"{state_name} · {elapsed/60:.1f}min{stats_str}{stale_str}"))

        if state_name != last_state:
            # On any state transition, dump the full Google-reported snapshot —
            # same info as the debug script, so users don't need to drop to a
            # Python shell to introspect a batch.
            ct = getattr(job, "create_time", None)
            st = getattr(job, "start_time", None)
            msgq.put(ProgressMsg(kind="log",
                text=f"[batch] state={state_name} (poll #{poll_count}, {elapsed/60:.1f}min elapsed){stats_str}"))
            msgq.put(ProgressMsg(kind="log",
                text=f"[batch]   create_time={ct}  start_time={st}  update_time={up}"))
            last_state = state_name
        elif poll_count % 5 == 0:
            msgq.put(ProgressMsg(kind="log",
                text=f"[batch] still {state_name} (poll #{poll_count}, {elapsed/60:.1f}min elapsed){stats_str}{stale_str}"))

        # Nudge if PENDING for a long time — not a bug, just a slow queue
        if (state_name == "JOB_STATE_PENDING" and elapsed > 600 and not pending_note_shown):
            msgq.put(ProgressMsg(kind="log",
                text=("[batch] PENDING for 10+ minutes — this is Google's server-side queue, not a client bug. "
                      "New accounts, free tier, or peak-hour submissions can wait 10-60+ minutes before "
                      "flipping to RUNNING. The app can be closed and reopened; polling resumes.")))
            pending_note_shown = True

        # Separate nudge for RUNNING-but-silent: likely stuck, suggest cancel
        if (state_name == "JOB_STATE_RUNNING" and stale_str and elapsed > 900 and not pending_note_shown):
            msgq.put(ProgressMsg(kind="log",
                text=(f"[batch] RUNNING but server hasn't updated in {stale_sec/60:.0f}min and no completion_stats. "
                      f"This CAN indicate a wedged batch. Consider cancelling (via Python: "
                      f"client.batches.cancel(name='{batch_name}')) and resubmitting.")))
            pending_note_shown = True

        if state_name in BATCH_COMPLETED_STATES:
            break

        for _ in range(BATCH_POLL_SECONDS):
            if stop_event.is_set():
                break
            time.sleep(1)

    if stop_event.is_set() and job and getattr(job.state, "name", "") not in BATCH_COMPLETED_STATES:
        msgq.put(ProgressMsg(kind="log", text="poll stopped by user — batch still running on server"))
        msgq.put(ProgressMsg(kind="done"))
        return

    final_state = getattr(job.state, "name", "unknown") if job else "unknown"

    if final_state != "JOB_STATE_SUCCEEDED":
        msgq.put(ProgressMsg(kind="error", text=f"Batch finished in state: {final_state}"))
        # Still clean up any refs the submit left behind — batch isn't going to use them now.
        _cleanup_ref_files(engine, state.get("ref_file_names") or [])
        clear_batch_state(output_dir)
        msgq.put(ProgressMsg(kind="done"))
        return

    # Download results to a temp file, then stream-read line by line.
    # Avoids holding the full decoded result (500MB+ for a large batch) in memory alongside bytes.
    # Some batch result file names exceed the Files API 40-char limit
    # (python-genai issue #1759) — if the SDK's `files.download` fails with
    # that class of error, fall back to a direct HTTP GET using the URI.
    result_temp: Optional[Path] = None
    try:
        result_file = job.dest.file_name if job.dest else None
        if not result_file:
            raise RuntimeError("no dest file on completed batch")
        msgq.put(ProgressMsg(kind="log", text=f"Downloading results from {result_file}..."))
        result_bytes = _download_batch_result(engine, result_file, api_key, msgq)
        with tempfile.NamedTemporaryFile(
            prefix="forge_result_", suffix=".jsonl", delete=False, mode="wb",
        ) as tmpf:
            tmpf.write(result_bytes)
            result_temp = Path(tmpf.name)
        del result_bytes  # free ~N00MB immediately
    except Exception as e:
        msgq.put(ProgressMsg(kind="error", text=f"Result download failed: {e}"))
        msgq.put(ProgressMsg(kind="done"))
        return

    saved = 0
    refused = 0
    errors = 0
    already_present = 0
    manifests: dict[str, dict] = {}  # char_name -> manifest dict

    def _iter_result_lines():
        with open(result_temp, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                yield line

    for line in _iter_result_lines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except Exception:
            errors += 1
            continue
        key = entry.get("key")
        info = key_map.get(key)
        if not info:
            continue
        char_name = info["char"]
        slot = info["slot"]
        flat = info["flat"]
        category = info.get("category")
        caption = info.get("caption", "")

        out_char = output_dir / char_name
        out_img = out_char / f"{slot:03d}.png"
        out_txt = out_char / f"{slot:03d}.txt"

        # Lazily load per-char manifests for incremental updates
        if char_name not in manifests:
            manifests[char_name] = load_manifest(out_char)

        # Skip if a prior partially-processed resume already wrote this slot.
        # Preserves any validation scores or manual edits that happened since.
        if _slot_file_complete(out_char, slot):
            already_present += 1
            continue

        resp = entry.get("response") or {}
        candidates = resp.get("candidates") or []
        image_data = None
        refusal_text = ""
        for cand in candidates:
            content = cand.get("content") or {}
            for part in (content.get("parts") or []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    image_data = inline["data"]
                    break
                if part.get("text"):
                    refusal_text = (refusal_text + " | " + part["text"]).strip(" |")
            if image_data:
                break
            fr = cand.get("finishReason") or cand.get("finish_reason")
            if fr:
                refusal_text = (refusal_text + f" finish={fr}").strip()
            # Surface which specific safety categories blocked the request —
            # 'finish=SAFETY' alone tells the user nothing actionable. The
            # category name (e.g. HARM_CATEGORY_SEXUALLY_EXPLICIT) at least
            # tells them which prompt direction to avoid.
            for rating in (cand.get("safetyRatings") or []):
                cat = rating.get("category") or ""
                prob = rating.get("probability") or ""
                blocked = rating.get("blocked", False)
                if blocked or prob in ("HIGH", "MEDIUM"):
                    refusal_text = (refusal_text + f" {cat}={prob}").strip()
        # Also check top-level promptFeedback for block reasons
        pf = resp.get("promptFeedback") or {}
        pf_block = pf.get("blockReason") or pf.get("block_reason")
        if pf_block and not image_data:
            refusal_text = (refusal_text + f" promptBlock={pf_block}").strip()

        if image_data:
            try:
                from io import BytesIO
                img_bytes = base64.b64decode(image_data)
                with Image.open(BytesIO(img_bytes)) as im:
                    im.save(out_img, format="PNG")
                # If verify_captions was on at submit time, re-caption via vision
                # model now using the actual generated image. This keeps batch
                # mode behaviorally consistent with sync mode's verify toggle.
                # Trigger is stored explicitly in key_map — don't parse from
                # the templated caption (fragile: trigger could contain commas).
                final_caption = caption
                if state.get("verify_captions"):
                    try:
                        trigger = info.get("trigger") or char_name
                        new_cap = engine.caption_image(img_bytes, trigger)
                        if new_cap and len(new_cap) > 20:
                            final_caption = new_cap
                    except Exception:
                        pass  # fall back to templated caption
                out_txt.write_text(final_caption, encoding="utf-8")
                extra = {"category": category} if category else {}
                record_slot(manifests[char_name], slot, flat, kept=True, **extra)
                saved += 1
            except Exception as e:
                errors += 1
                msgq.put(ProgressMsg(kind="error", char=char_name,
                                     text=f"[{char_name}] {slot:03d} save failed: {e}"))
        else:
            # Don't stomp a successfully-saved slot's manifest entry with a
            # refusal from a stale or duplicate batch result. If the PNG+TXT
            # are already present and valid on disk, preserve the existing
            # kept=True record.
            if _slot_file_complete(out_char, slot):
                already_present += 1
                continue
            refused += 1
            extra = {"refused": True, "refuse_reason": (refusal_text or "no image")[:200]}
            if category:
                extra["category"] = category
            record_slot(manifests[char_name], slot, flat, kept=False, **extra)

        msgq.put(ProgressMsg(kind="progress", char=char_name,
                             current=slot, total=0))

    # Flush manifests
    for char_name, manifest in manifests.items():
        try:
            save_manifest(output_dir / char_name, manifest)
        except Exception as e:
            msgq.put(ProgressMsg(kind="error", text=f"manifest save failed for {char_name}: {e}"))

    summary = f"Batch complete — saved {saved}, refused {refused}, errors {errors}"
    if already_present:
        summary += f", skipped {already_present} (already present)"
    msgq.put(ProgressMsg(kind="log", text=summary))

    # Clean up the temp result file we streamed from
    if result_temp:
        try:
            result_temp.unlink(missing_ok=True)
        except Exception:
            pass

    # Clean up any Files API uploads (refs) to keep the account tidy.
    # Google auto-expires in 48h regardless but tidy is better than littered.
    deleted = _cleanup_ref_files(engine, state.get("ref_file_names") or [])
    if deleted:
        msgq.put(ProgressMsg(kind="log", text=f"cleaned up {deleted} Files API ref(s)"))

    clear_batch_state(output_dir)
    msgq.put(ProgressMsg(kind="done"))


# -------------------------------------------------------------------------
# Custom widgets
# -------------------------------------------------------------------------

def _bind_wheel_on_hover(canvas: tk.Canvas, *also_bind_enter_leave_on: tk.Widget):
    """Bind MouseWheel to `canvas` only while the cursor is over it or any of
    the extra widgets. Uses dynamic bind_all + unbind_all on Enter/Leave so
    multiple scrollable windows don't clobber each other's wheel bindings.
    """
    def _on_wheel(e):
        if canvas.winfo_exists():
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

    def _enter(_e):
        canvas.bind_all("<MouseWheel>", _on_wheel)

    def _leave(_e):
        canvas.unbind_all("<MouseWheel>")

    canvas.bind("<Enter>", _enter)
    canvas.bind("<Leave>", _leave)
    for w in also_bind_enter_leave_on:
        w.bind("<Enter>", _enter)
        w.bind("<Leave>", _leave)


def apply_dark_titlebar(root: tk.Tk):
    """Force dark title bar on Windows 11."""
    if sys.platform != "win32":
        return
    try:
        root.update_idletasks()
        hwnd = ctypes.windll.user32.GetParent(root.winfo_id())
        value = ctypes.c_int(1)
        for attr in (20, 19):  # 20 = Win11+, 19 = older Win10 builds
            ctypes.windll.dwmapi.DwmSetWindowAttribute(
                hwnd, attr, ctypes.byref(value), ctypes.sizeof(value)
            )
    except Exception:
        pass


class HoverButton(tk.Label):
    """Flat custom button with proper hover/press states. Looks modern on dark bg."""

    def __init__(self, parent, text, command=None, *, variant="secondary", width=None, pad=(14, 8), font=None):
        self.variant = variant
        self._cmd = command
        self._enabled = True
        self._colors = self._palette(variant)

        super().__init__(
            parent, text=text,
            bg=self._colors["bg"], fg=self._colors["fg"],
            activebackground=self._colors["active"],
            padx=pad[0], pady=pad[1],
            bd=0, highlightthickness=0, cursor="hand2",
            font=font or (FONT_UI, 10, "bold" if variant == "primary" else "normal"),
        )
        if width:
            self.configure(width=width)

        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    @staticmethod
    def _palette(variant):
        if variant == "primary":
            return {"bg": COLOR["accent"], "fg": "#1a0e08", "hover": COLOR["accent_hi"],
                    "active": COLOR["accent_lo"], "disabled": "#5a3e32"}
        if variant == "danger":
            return {"bg": COLOR["surface2"], "fg": COLOR["text_dim"], "hover": "#2a2f3a",
                    "active": "#1a1e26", "disabled": "#20242d"}
        if variant == "ghost":
            return {"bg": COLOR["bg"], "fg": COLOR["text_dim"], "hover": COLOR["surface"],
                    "active": COLOR["surface2"], "disabled": COLOR["bg"]}
        return {"bg": COLOR["surface2"], "fg": COLOR["text"], "hover": "#262b35",
                "active": "#161a20", "disabled": "#16191f"}

    def _on_enter(self, _e):
        if self._enabled:
            self.configure(bg=self._colors["hover"])

    def _on_leave(self, _e):
        if self._enabled:
            self.configure(bg=self._colors["bg"])

    def _on_press(self, _e):
        if self._enabled:
            self.configure(bg=self._colors["active"])

    def _on_release(self, _e):
        if not self._enabled:
            return
        self.configure(bg=self._colors["hover"])
        if self._cmd:
            self._cmd()

    def set_enabled(self, val: bool):
        self._enabled = val
        self.configure(
            bg=self._colors["bg"] if val else self._colors["disabled"],
            fg=self._colors["fg"] if val else COLOR["muted"],
            cursor="hand2" if val else "",
        )


class Toggle(tk.Label):
    """Click-to-toggle checkbox with unicode glyph. Legible on dark bg."""

    def __init__(self, parent, variable: tk.BooleanVar, *, size=13, bg=None):
        self._var = variable
        self._bg = bg or COLOR["row"]
        super().__init__(
            parent, text="", bg=self._bg, fg=COLOR["accent"],
            bd=0, highlightthickness=0, cursor="hand2",
            font=(FONT_UI, size),
        )
        self._redraw()
        self.bind("<Button-1>", self._toggle)
        variable.trace_add("write", lambda *_: self._redraw())

    def _redraw(self):
        on = bool(self._var.get())
        self.configure(
            text="◉" if on else "○",
            fg=COLOR["accent"] if on else COLOR["muted"],
        )

    def _toggle(self, _e):
        self._var.set(not self._var.get())


class StatusPill(tk.Label):
    """Colored status pill shown at row end."""

    STATES = {
        "idle":    (COLOR["muted"],  "idle"),
        "queued":  (COLOR["info"],   "queued"),
        "running": (COLOR["warn"],   "running…"),
        "done":    (COLOR["ok"],     "done ✓"),
        "error":   (COLOR["err"],    "error"),
    }

    def __init__(self, parent, *, bg=None):
        super().__init__(
            parent, text="—",
            bg=bg or COLOR["row"], fg=COLOR["muted"],
            bd=0, highlightthickness=0,
            font=(FONT_MONO, 9), anchor="e", width=14,
        )

    def set_state(self, state: str, text: Optional[str] = None):
        fg, default = self.STATES.get(state, (COLOR["muted"], "—"))
        self.configure(fg=fg, text=text or default)


class FilePicker(tk.Frame):
    """File cell: ellipsized filename + compact 'edit' button."""

    def __init__(self, parent, initial: Path, on_pick, *, bg=None, on_change=None):
        super().__init__(parent, bg=bg or COLOR["row"])
        self._bg = bg or COLOR["row"]
        self._on_pick = on_pick
        self._on_change = on_change
        self._path = initial

        self.label = tk.Label(
            self, text="", bg=COLOR["surface"], fg=COLOR["text_dim"],
            bd=0, highlightthickness=0, anchor="w",
            font=(FONT_MONO, 9), padx=10, pady=6,
            width=30,  # fixed char width → identical cell size on every row
        )
        self.label.pack(side="left", fill="x", expand=True)

        self.btn = tk.Label(
            self, text="edit", bg=COLOR["surface2"], fg=COLOR["text_dim"],
            bd=0, highlightthickness=0, cursor="hand2",
            font=(FONT_UI, 9), padx=10, pady=6,
        )
        self.btn.pack(side="left", padx=(3, 0))
        self.btn.bind("<Enter>", lambda e: self.btn.configure(bg="#262b35", fg=COLOR["text"]))
        self.btn.bind("<Leave>", lambda e: self.btn.configure(bg=COLOR["surface2"], fg=COLOR["text_dim"]))
        self.btn.bind("<Button-1>", lambda e: self._pick())

        self._set_path(initial)

    def set_path(self, p: Path):
        """Public API — updates the displayed filename and fires on_change."""
        self._path = p
        self.label.configure(text=self._ellipsize(p.name, 30))
        if self._on_change:
            try:
                self._on_change()
            except Exception:
                pass

    # Backwards-compat shim — kept to avoid breaking external callers.
    _set_path = set_path

    def _pick(self):
        new = self._on_pick(self._path)
        if new:
            self.set_path(Path(new))

    @staticmethod
    def _ellipsize(s: str, maxlen: int) -> str:
        if len(s) <= maxlen:
            return s
        keep_end = 10
        keep_start = maxlen - keep_end - 1
        return s[:keep_start] + "…" + s[-keep_end:]

    @property
    def path(self) -> Path:
        return self._path


class Card(tk.Frame):
    """Rounded-looking panel (tk can't do true radii — we use border + padding)."""

    def __init__(self, parent, *, padx=18, pady=16):
        super().__init__(
            parent, bg=COLOR["surface"],
            bd=0, highlightthickness=1,
            highlightbackground=COLOR["border"],
            highlightcolor=COLOR["border"],
        )
        self.inner = tk.Frame(self, bg=COLOR["surface"])
        self.inner.pack(fill="both", expand=True, padx=padx, pady=pady)


# -------------------------------------------------------------------------
# Character row
# -------------------------------------------------------------------------

class CharRow(tk.Frame):
    def __init__(self, parent, char: Character, index: int,
                 on_pick_face, on_pick_body, on_preview, on_review, on_edit_dist,
                 on_any_change=None):
        bg = COLOR["row"] if index % 2 == 0 else COLOR["row_alt"]
        super().__init__(parent, bg=bg, bd=0, highlightthickness=0)
        self.char = char
        self._bg = bg
        self._on_preview = on_preview
        self._on_any_change = on_any_change

        for col, w in enumerate(COL_WIDTHS):
            self.grid_columnconfigure(col, minsize=w, weight=0)

        # col 0 — enable toggle
        self.enabled_var = tk.BooleanVar(value=char.enabled)
        Toggle(self, self.enabled_var, bg=bg).grid(row=0, column=0, padx=(COL_PAD_X, 4), pady=10)

        # col 1 — character name
        tk.Label(
            self, text=char.name, bg=bg, fg=COLOR["text"],
            font=(FONT_UI, 10, "bold"), anchor="w",
        ).grid(row=0, column=1, sticky="w", padx=COL_PAD_X)

        # col 2 — face picker
        self.face_picker = FilePicker(self, char.face,
                                      on_pick=lambda cur: on_pick_face(self), bg=bg,
                                      on_change=self._fire_change)
        self.face_picker.grid(row=0, column=2, sticky="ew", padx=COL_PAD_X, pady=6)

        # col 3 — body picker
        self.body_picker = FilePicker(self, char.body,
                                      on_pick=lambda cur: on_pick_body(self), bg=bg,
                                      on_change=self._fire_change)
        self.body_picker.grid(row=0, column=3, sticky="ew", padx=COL_PAD_X, pady=6)

        # col 4 — trigger word
        self.trigger_var = tk.StringVar(value=char.trigger)
        trig_wrap = tk.Frame(self, bg=COLOR["surface"], bd=0, highlightthickness=1,
                             highlightbackground=COLOR["border"])
        trig_wrap.grid(row=0, column=4, sticky="w", padx=COL_PAD_X, pady=6)
        tk.Entry(
            trig_wrap, textvariable=self.trigger_var,
            bg=COLOR["surface"], fg=COLOR["accent"],
            insertbackground=COLOR["text"], bd=0, relief="flat",
            width=16, font=(FONT_MONO, 10),
        ).pack(padx=10, pady=6)

        # col 5 — count
        self.count_var = tk.IntVar(value=char.count)
        cnt_wrap = tk.Frame(self, bg=COLOR["surface"], bd=0, highlightthickness=1,
                            highlightbackground=COLOR["border"])
        cnt_wrap.grid(row=0, column=5, sticky="w", padx=COL_PAD_X, pady=6)
        tk.Spinbox(
            cnt_wrap, from_=1, to=500, textvariable=self.count_var, width=5,
            bg=COLOR["surface"], fg=COLOR["text"],
            buttonbackground=COLOR["surface2"],
            bd=0, relief="flat", font=(FONT_MONO, 10),
            insertbackground=COLOR["text"],
        ).pack(padx=6, pady=4)

        # col 6 — variation toggle + label
        vary_frame = tk.Frame(self, bg=bg)
        vary_frame.grid(row=0, column=6, sticky="w", padx=COL_PAD_X)
        self.vary_var = tk.BooleanVar(value=char.vary_outfit)
        Toggle(vary_frame, self.vary_var, bg=bg).pack(side="left")
        tk.Label(vary_frame, text=" vary outfit", bg=bg, fg=COLOR["text_dim"],
                 font=(FONT_UI, 9)).pack(side="left")

        # col 7 — preview + review + dist (side by side)
        actions_frame = tk.Frame(self, bg=bg)
        actions_frame.grid(row=0, column=7, padx=COL_PAD_X, pady=8, sticky="w")
        HoverButton(actions_frame, "preview", command=lambda: on_preview(self),
                    variant="ghost", pad=(10, 6)).pack(side="left", padx=(0, 4))
        HoverButton(actions_frame, "review", command=lambda: on_review(self),
                    variant="secondary", pad=(10, 6)).pack(side="left", padx=(0, 4))
        self.dist_btn = HoverButton(actions_frame, "dist",
                                    command=lambda: on_edit_dist(self),
                                    variant="ghost", pad=(10, 6))
        self.dist_btn.pack(side="left")

        # col 8 — status pill
        self.status = StatusPill(self, bg=bg)
        self.status.set_state("idle")
        self.status.grid(row=0, column=8, sticky="e", padx=(COL_PAD_X, COL_PAD_X + 4))

        # subtle divider
        tk.Frame(self, bg=COLOR["divider"], height=1).grid(
            row=1, column=0, columnspan=len(COL_WIDTHS), sticky="ew"
        )

        self._refresh_dist_button()

        # Bind change traces for auto-save (after all vars exist)
        for var in (self.enabled_var, self.trigger_var, self.count_var, self.vary_var):
            try:
                var.trace_add("write", lambda *_: self._fire_change())
            except tk.TclError:
                pass

    def _fire_change(self):
        if self._on_any_change:
            try:
                self._on_any_change()
            except Exception:
                pass

    def apply_back(self) -> Character:
        self.char.enabled = self.enabled_var.get()
        self.char.trigger = self.trigger_var.get().strip() or self.char.trigger
        new_count = int(self.count_var.get())
        self.char.count = new_count
        # If user manually changed count and distribution is set, clear distribution
        # (distribution total must match count, but editing count disables that coupling)
        if self.char.distribution and sum(self.char.distribution.values()) != new_count:
            self.char.distribution = None
            self._refresh_dist_button()
        self.char.vary_outfit = self.vary_var.get()
        self.char.face = self.face_picker.path
        self.char.body = self.body_picker.path
        return self.char

    def _refresh_dist_button(self):
        """Visual indicator on the dist button: 'dist' when random, accent label when custom."""
        if not hasattr(self, "dist_btn"):
            return
        if self.char.distribution:
            d = self.char.distribution
            compact = f"{d.get('close',0)}·{d.get('mid',0)}·{d.get('full',0)}·{d.get('random',0)}"
            self.dist_btn.configure(text=f"dist  {compact}")
            self.dist_btn._colors = HoverButton._palette("primary")
            self.dist_btn.configure(bg=self.dist_btn._colors["bg"],
                                    fg=self.dist_btn._colors["fg"])
        else:
            self.dist_btn.configure(text="dist")
            self.dist_btn._colors = HoverButton._palette("ghost")
            self.dist_btn.configure(bg=self.dist_btn._colors["bg"],
                                    fg=self.dist_btn._colors["fg"])


# -------------------------------------------------------------------------
# Distribution editor — per-character framing mix
# -------------------------------------------------------------------------

class DistributionEditor:
    """Popup to set how a character's `count` images split across framing categories."""

    CATEGORY_LABELS = [
        ("close", "Close  (face · bust)",         "4 framings: tight close-up, extreme close-up, close-up, bust"),
        ("mid",   "Mid    (waist-up · half-body)","3 framings: waist-up, half-body, medium from hips"),
        ("full",  "Full   (three-quarter · full)","3 framings: three-quarter body, full body, full body wide"),
        ("random","Random (any framing)",         "samples any of the 10 framings uniformly"),
    ]

    def __init__(self, parent, row: "CharRow"):
        self.row = row
        self.char = row.char
        self.win = tk.Toplevel(parent)
        self.win.title(f"Distribution — {self.char.name}")
        self.win.configure(bg=COLOR["bg"])
        self.win.geometry("640x500")
        self.win.resizable(False, False)
        apply_dark_titlebar(self.win)
        self.win.transient(parent)
        self.win.grab_set()

        current = self.char.distribution or {}
        self._vars = {}
        for cat, _, _ in self.CATEGORY_LABELS:
            self._vars[cat] = tk.IntVar(value=int(current.get(cat, 0)))
        self._total_var = tk.StringVar()

        self._build_ui()
        self._update_total()

    def _build_ui(self):
        hdr = tk.Frame(self.win, bg=COLOR["bg"])
        hdr.pack(fill="x", padx=24, pady=(20, 4))
        tk.Label(hdr, text=f"Distribution — {self.char.name}",
                 bg=COLOR["bg"], fg=COLOR["text"],
                 font=(FONT_UI, 15, "bold")).pack(anchor="w")
        tk.Label(
            self.win,
            text=("split the generation across explicit buckets. set a bucket to 0 to skip it. "
                  "slots get assigned in category order (close → mid → full → random), "
                  "so with 10·0·10·10 slots 1-10 are close-ups, 11-20 are full-body, 21-30 are random."),
            bg=COLOR["bg"], fg=COLOR["muted"],
            font=(FONT_UI, 9), wraplength=580,
            justify="left", anchor="w",
        ).pack(fill="x", padx=24, pady=(0, 14))

        card = tk.Frame(self.win, bg=COLOR["surface"], bd=0,
                        highlightthickness=1, highlightbackground=COLOR["border"])
        card.pack(fill="x", padx=24, pady=(0, 12))

        for cat, label, hint in self.CATEGORY_LABELS:
            row = tk.Frame(card, bg=COLOR["surface"])
            row.pack(fill="x", padx=16, pady=8)
            tk.Label(row, text=label, bg=COLOR["surface"], fg=COLOR["text"],
                     font=(FONT_UI, 10, "bold"), width=32, anchor="w").pack(side="left")

            sp_wrap = tk.Frame(row, bg=COLOR["bg"], bd=0,
                               highlightthickness=1, highlightbackground=COLOR["border"])
            sp_wrap.pack(side="left", padx=(0, 12))
            sp = tk.Spinbox(
                sp_wrap, from_=0, to=500, textvariable=self._vars[cat], width=5,
                bg=COLOR["bg"], fg=COLOR["text"],
                buttonbackground=COLOR["surface2"],
                bd=0, relief="flat", font=(FONT_MONO, 11),
                insertbackground=COLOR["text"],
                command=self._update_total,
            )
            sp.pack(padx=6, pady=4)
            # Trace too, in case user types directly
            try:
                self._vars[cat].trace_add("write", lambda *_: self._update_total())
            except tk.TclError:
                pass

            tk.Label(row, text=hint, bg=COLOR["surface"], fg=COLOR["muted"],
                     font=(FONT_UI, 9), anchor="w").pack(side="left", fill="x", expand=True)

        # Total
        total_row = tk.Frame(self.win, bg=COLOR["bg"])
        total_row.pack(fill="x", padx=24, pady=(4, 18))
        tk.Label(total_row, text="TOTAL IMAGES", bg=COLOR["bg"], fg=COLOR["muted"],
                 font=(FONT_UI, 9, "bold")).pack(side="left")
        tk.Label(total_row, textvariable=self._total_var, bg=COLOR["bg"],
                 fg=COLOR["accent"], font=(FONT_MONO, 14, "bold")).pack(side="left", padx=12)

        # Buttons
        btn_row = tk.Frame(self.win, bg=COLOR["bg"])
        btn_row.pack(fill="x", padx=24, pady=(0, 22))

        HoverButton(btn_row, "clear (use random)", command=self._clear,
                    variant="ghost", pad=(14, 8)).pack(side="left")

        HoverButton(btn_row, "cancel", command=self.win.destroy,
                    variant="ghost", pad=(14, 8)).pack(side="right", padx=(6, 0))
        HoverButton(btn_row, "  apply  ", command=self._apply,
                    variant="primary", pad=(18, 9),
                    font=(FONT_UI, 10, "bold")).pack(side="right", padx=6)

    def _update_total(self):
        try:
            t = sum(int(v.get()) for v in self._vars.values())
        except (tk.TclError, ValueError):
            t = 0
        self._total_var.set(str(t))

    def _clear(self):
        self.char.distribution = None
        self.row._refresh_dist_button()
        # Explicitly notify — no Tk var changed so the trace won't fire.
        self.row._fire_change()
        self.win.destroy()

    def _apply(self):
        dist = {cat: max(0, int(v.get())) for cat, v in self._vars.items()}
        total = sum(dist.values())
        if total == 0:
            messagebox.showwarning("empty distribution",
                                   "at least one bucket must be > 0, or use 'clear (use random)'")
            return
        self.char.distribution = dist
        self.char.count = total
        self.row.count_var.set(total)
        self.row._refresh_dist_button()
        self.win.destroy()


# -------------------------------------------------------------------------
# Prompt preview — shows rendered prompt + caption for the first few planned specs
# -------------------------------------------------------------------------

class PromptPreviewWindow:
    def __init__(self, parent, char: Character):
        self.char = char
        self.win = tk.Toplevel(parent)
        self.win.title(f"Prompt preview — {char.name}")
        self.win.configure(bg=COLOR["bg"])
        self.win.geometry("1100x860")
        self.win.minsize(900, 600)
        apply_dark_titlebar(self.win)
        self._build_ui()

    def _build_ui(self):
        hdr = tk.Frame(self.win, bg=COLOR["bg"])
        hdr.pack(fill="x", padx=22, pady=(18, 4))
        tk.Label(hdr, text=f"Prompt preview — {self.char.name}",
                 bg=COLOR["bg"], fg=COLOR["text"],
                 font=(FONT_UI, 17, "bold")).pack(side="left")
        tk.Label(hdr, text=f"   trigger={self.char.trigger}  ·  vary_outfit={self.char.vary_outfit}",
                 bg=COLOR["bg"], fg=COLOR["muted"],
                 font=(FONT_MONO, 10)).pack(side="left", pady=(6, 0))

        tk.Label(
            self.win,
            text=("these are the first three generation specs planned for this character. "
                  "each run is deterministic per-trigger, so what you see here is what actually gets sent."),
            bg=COLOR["bg"], fg=COLOR["muted"],
            font=(FONT_UI, 9), wraplength=1050,
            justify="left", anchor="w",
        ).pack(fill="x", padx=22, pady=(0, 10))

        # Scrollable body
        outer = tk.Frame(self.win, bg=COLOR["surface"], bd=0,
                         highlightthickness=1, highlightbackground=COLOR["border"])
        outer.pack(fill="both", expand=True, padx=22, pady=(0, 16))

        canvas = tk.Canvas(outer, bg=COLOR["surface"], highlightthickness=0, bd=0)
        canvas.pack(side="left", fill="both", expand=True)
        vs = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview,
                           style="Forge.Vertical.TScrollbar")
        vs.pack(side="right", fill="y")
        canvas.configure(yscrollcommand=vs.set)

        inner = tk.Frame(canvas, bg=COLOR["surface"])
        win = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(win, width=e.width))
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        _bind_wheel_on_hover(canvas, inner, self.win)

        # Aspect distribution across all framings
        self._aspect_table(inner)

        # System prompt block
        self._section(inner, "SYSTEM PROMPT (sent on every call)", P.SYSTEM_PROMPT,
                      color=COLOR["text_dim"])

        # Planned specs — if a distribution is set, show one per non-zero category.
        # Otherwise show the first 3 from the random shuffle.
        if self.char.distribution:
            tk.Label(
                inner,
                text=("this character uses a custom distribution: "
                      + ", ".join(f"{cat}={self.char.distribution.get(cat, 0)}"
                                  for cat in P.CATEGORY_ORDER
                                  if self.char.distribution.get(cat, 0))),
                bg=COLOR["surface"], fg=COLOR["accent"],
                font=(FONT_UI, 9, "bold"),
                anchor="w", justify="left", wraplength=1050,
            ).pack(fill="x", padx=22, pady=(8, 0))

            for cat in P.CATEGORY_ORDER:
                if self.char.distribution.get(cat, 0) <= 0:
                    continue
                cat_dist = {c: 0 for c in P.CATEGORY_ORDER}
                cat_dist[cat] = 1
                cat_sample = P.plan_jobs_distributed(
                    cat_dist, vary_outfit=self.char.vary_outfit, trigger=self.char.trigger)
                if cat_sample:
                    flat, spec, outfit, _ = cat_sample[0]
                    self._spec_section(inner, f"{cat.upper()} sample", flat, spec, outfit)
        else:
            jobs = P.plan_jobs(3, vary_outfit=self.char.vary_outfit, trigger=self.char.trigger)
            for idx, (flat, spec, outfit) in enumerate(jobs, start=1):
                self._spec_section(inner, idx, flat, spec, outfit)

        # Close button
        tb = tk.Frame(self.win, bg=COLOR["bg"])
        tb.pack(fill="x", padx=22, pady=(0, 18))
        HoverButton(tb, "close", command=self.win.destroy,
                    variant="secondary", pad=(16, 8)).pack(side="right")

    def _aspect_table(self, parent):
        from collections import Counter
        aspects = [P.smart_aspect(f) for f in P.FRAMINGS]
        dist = Counter(aspects)

        hdr = tk.Frame(parent, bg=COLOR["surface"])
        hdr.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(hdr, text="SMART ASPECT RATIO MAPPING",
                 bg=COLOR["surface"], fg=COLOR["muted"],
                 font=(FONT_UI, 9, "bold")).pack(anchor="w")

        tbl = tk.Frame(parent, bg=COLOR["bg"], bd=0,
                       highlightthickness=1, highlightbackground=COLOR["border"])
        tbl.pack(fill="x", padx=18, pady=(4, 4))

        for framing in P.FRAMINGS:
            asp = P.smart_aspect(framing)
            row = tk.Frame(tbl, bg=COLOR["bg"])
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=framing, bg=COLOR["bg"], fg=COLOR["text_dim"],
                     font=(FONT_MONO, 9), anchor="w", width=50).pack(side="left")
            tk.Label(row, text="→", bg=COLOR["bg"], fg=COLOR["muted"],
                     font=(FONT_MONO, 9)).pack(side="left", padx=6)
            tk.Label(row, text=asp, bg=COLOR["bg"], fg=COLOR["accent"],
                     font=(FONT_MONO, 10, "bold"), anchor="w").pack(side="left")

        # Distribution summary
        summary = "  ·  ".join(f"{asp} ≈ {n*10}%" for asp, n in sorted(dist.items(), key=lambda x: -x[1]))
        tk.Label(parent, text=f"distribution over {len(P.FRAMINGS)} framings:  {summary}",
                 bg=COLOR["surface"], fg=COLOR["muted"],
                 font=(FONT_UI, 9, "italic")).pack(anchor="w", padx=22, pady=(0, 6))

    def _section(self, parent, title, body, color=None):
        wrap = tk.Frame(parent, bg=COLOR["surface"])
        wrap.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(wrap, text=title, bg=COLOR["surface"], fg=COLOR["muted"],
                 font=(FONT_UI, 9, "bold")).pack(anchor="w")
        body_frame = tk.Frame(parent, bg=COLOR["bg"], bd=0,
                              highlightthickness=1, highlightbackground=COLOR["border"])
        body_frame.pack(fill="x", padx=18, pady=(4, 8))
        tk.Label(
            body_frame, text=body, bg=COLOR["bg"], fg=color or COLOR["text"],
            font=(FONT_MONO, 9), justify="left", anchor="w", wraplength=1000,
            padx=14, pady=10,
        ).pack(fill="x")

    def _spec_section(self, parent, idx, flat, spec, outfit):
        aspect = P.smart_aspect(spec["framing"])
        label = f"SLOT #{idx:03d}" if isinstance(idx, int) else str(idx)
        hdr = tk.Frame(parent, bg=COLOR["surface"])
        hdr.pack(fill="x", padx=18, pady=(18, 0))
        tk.Label(hdr, text=f"{label}  ·  flat_index={flat}  ·  aspect={aspect}",
                 bg=COLOR["surface"], fg=COLOR["accent"],
                 font=(FONT_UI, 10, "bold")).pack(anchor="w")

        # Spec table
        tbl = tk.Frame(parent, bg=COLOR["bg"], bd=0,
                       highlightthickness=1, highlightbackground=COLOR["border"])
        tbl.pack(fill="x", padx=18, pady=(4, 4))
        fields = [
            ("framing", spec["framing"]),
            ("angle", spec["angle"]),
            ("expression", spec["expression"]),
            ("pose", spec["pose"]),
            ("environment", spec["environment"]),
            ("lighting", spec["lighting"]),
            ("weather", f"{spec['weather']}  ({'outdoor' if spec.get('outdoor') else 'indoor — through the window'})"),
            ("prop", spec.get("prop") or "(none)"),
            ("outfit", outfit or "(locked to body ref)"),
        ]
        for k, v in fields:
            row = tk.Frame(tbl, bg=COLOR["bg"])
            row.pack(fill="x", padx=10, pady=2)
            tk.Label(row, text=k, bg=COLOR["bg"], fg=COLOR["muted"],
                     font=(FONT_UI, 9, "bold"), width=14, anchor="w").pack(side="left")
            tk.Label(row, text=v, bg=COLOR["bg"], fg=COLOR["text_dim"],
                     font=(FONT_MONO, 9), anchor="w", justify="left",
                     wraplength=900).pack(side="left", fill="x", expand=True)

        # Rendered prompt + caption
        self._section(parent, "RENDERED PROMPT (user message sent with refs)",
                      P.build_prompt_text(spec, self.char.trigger, outfit))
        self._section(parent, "RENDERED CAPTION (saved to .txt)",
                      P.build_caption(spec, self.char.trigger, outfit))


# -------------------------------------------------------------------------
# Trainer config export
# -------------------------------------------------------------------------

def _aitoolkit_full_config(char: Character, char_dir: Path) -> str:
    """Render a complete, runnable ai-toolkit training config YAML for one
    character. Based on ostris' official train_lora_qwen_image_24gb.yaml
    template, tuned for 32GB VRAM (RTX 5090 / A6000 Ada):
      - rank bumped 16 → 32
      - batch_size bumped 1 → 2
      - resolution bucket bumped to [768, 1024, 1344]
    Base model stays quantized (uint3) — Qwen-Image is too big to fit
    un-quantized even in 32GB alongside LoRA + optimizer state.
    """
    trigger = char.trigger
    # Validation prompts — use the trigger so you can eyeball the LoRA mid-training
    sample_prompts = [
        f"{trigger}, close-up portrait, soft natural smile, bright window light, neutral studio backdrop",
        f"{trigger}, full body shot, standing in a park, golden hour sunlight",
        f"{trigger}, three-quarter body shot, leather jacket and jeans, city street at dusk",
        f"{trigger}, waist-up shot, casual sweater, cozy café interior, warm lamp light",
        f"{trigger}, editorial portrait, confident expression, moody studio lighting",
        f"{trigger}, laughing, outdoor summer afternoon, candid snapshot",
        f"{trigger}, looking over the shoulder toward camera, elegant evening dress",
        f"{trigger}, sitting at a wooden café table holding a mug, soft morning light",
    ]
    indent = "  "
    prompt_block = "\n".join(f'{indent*5}- "{p}"' for p in sample_prompts)

    return f"""# ai-toolkit Qwen-Image LoRA training config for {char.name}
# generated by LoRA-Dataset-Forge
# trigger word: {trigger}
#
# Tuned for 32GB VRAM (RTX 5090 / A6000 Ada / etc).
# If you're on 24GB, drop `linear` + `linear_alpha` from 32 → 16 and
# `batch_size` from 2 → 1.
#
# Run with:   python run.py _aitoolkit_{char.name}.yaml

job: extension
config:
  name: "{trigger}_qwen_lora_v1"
  process:
    - type: 'sd_trainer'
      training_folder: "output/{trigger}"
      device: cuda:0
      trigger_word: "{trigger}"
      network:
        type: "lora"
        linear: 32
        linear_alpha: 32
      save:
        dtype: float16
        save_every: 250
        max_step_saves_to_keep: 4
      datasets:
        - folder_path: "{char_dir.as_posix()}"
          caption_ext: "txt"
          caption_dropout_rate: 0.05
          shuffle_tokens: false
          cache_latents_to_disk: true
          resolution: [ 768, 1024, 1344 ]
      train:
        batch_size: 2
        cache_text_embeddings: true
        steps: 2500
        gradient_accumulation: 1
        train_unet: true
        train_text_encoder: false
        gradient_checkpointing: true
        noise_scheduler: "flowmatch"
        optimizer: "adamw8bit"
        lr: 1e-4
        dtype: bf16
      model:
        name_or_path: "Qwen/Qwen-Image"
        arch: "qwen_image"
        quantize: true
        qtype: "uint3|ostris/accuracy_recovery_adapters/qwen_image_torchao_uint3.safetensors"
        quantize_te: true
        qtype_te: "qfloat8"
        low_vram: false
      sample:
        sampler: "flowmatch"
        sample_every: 250
        width: 1024
        height: 1024
        prompts:
{prompt_block}
        neg: ""
        seed: 42
        walk_seed: true
        guidance_scale: 3
        sample_steps: 25
meta:
  name: "{trigger}"
  version: '1.0'
"""


def export_trainer_configs(chars: list[Character], output_dir: Path):
    """Generate a COMPLETE ai-toolkit YAML per character (primary target)
    plus a musubi-tuner TOML dataset block (alternate) and a training
    README. Each ai-toolkit config is directly runnable with
    `python run.py _aitoolkit_<char_name>.yaml`.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- musubi-tuner (alternate) — one combined TOML for all chars ----
    toml_lines = [
        "# musubi-tuner dataset config for Qwen-Image LoRA",
        "# generated by LoRA-Dataset-Forge",
        "",
        "[general]",
        'resolution = [1024, 1024]',
        'caption_extension = ".txt"',
        "batch_size = 1",
        "enable_bucket = true",
        "bucket_no_upscale = false",
        "",
    ]

    readme_lines = [
        "# LoRA Training — Dataset Ready",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M')}",
        f"Characters: {len(chars)}",
        "",
        "## Trigger words & generated configs",
        "",
    ]

    # ---- ai-toolkit (primary) — one COMPLETE config per character ----
    generated_configs: list[str] = []
    for char in chars:
        char_dir = (output_dir / char.name).resolve()
        char_path = char_dir.as_posix()

        yaml_text = _aitoolkit_full_config(char, char_dir)
        yaml_name = f"_aitoolkit_{char.name}.yaml"
        (output_dir / yaml_name).write_text(yaml_text, encoding="utf-8")
        generated_configs.append(yaml_name)

        # musubi block for this char
        toml_lines += [
            "[[datasets]]",
            f'image_directory = "{char_path}"',
            'caption_extension = ".txt"',
            'num_repeats = 2',
            "",
        ]

        readme_lines.append(
            f"- **{char.name}** → trigger `{char.trigger}` → "
            f"config `{yaml_name}` ({char_dir})"
        )

    readme_lines += [
        "",
        "## ai-toolkit by ostris (recommended)",
        "",
        "1. Install ai-toolkit: <https://github.com/ostris/ai-toolkit>",
        "2. Copy the generated `_aitoolkit_<char>.yaml` file(s) into your",
        "   ai-toolkit repo (any location works; the `folder_path` in the",
        "   YAML references this dataset dir by absolute path).",
        "3. Run each character's config:",
        "",
        "```bash",
        f"python run.py _aitoolkit_{chars[0].name if chars else 'CharName'}.yaml",
        "```",
        "",
        "Each YAML is a complete, runnable config — no template merging needed.",
        "",
        "### What's already set in each config",
        "",
        "- `name` uses the character's trigger word",
        "- `trigger_word` is set (ai-toolkit uses it to drop captions without the trigger)",
        "- `training_folder` is `output/<trigger>` (relative to where you run ai-toolkit)",
        "- `network: linear=32, linear_alpha=32` — reasonable rank for a 32GB card",
        "- `train: batch_size=2, steps=2500, lr=1e-4, optimizer=adamw8bit, dtype=bf16`",
        "- `model: Qwen/Qwen-Image` with `uint3` quantization on the transformer",
        "  (required — the un-quantized model doesn't fit on 32GB alongside LoRA state)",
        "- `sample: sample_every=250` with 8 validation prompts using your trigger",
        "- `datasets` points at the generated images for this character with",
        "  multi-res bucketing `[768, 1024, 1344]`",
        "",
        "### Common tweaks",
        "",
        "- **Underfitting after 2500 steps?** Bump `linear`/`linear_alpha` to 64",
        "  or push `steps` to 4000.",
        "- **Overfitting (LoRA memorizes a single image)?** Drop `linear` to 16,",
        "  bump `caption_dropout_rate` to 0.1, or cut `steps` to 1500.",
        "- **Using a 24GB card instead of 32GB?** Drop `linear` to 16, `batch_size`",
        "  to 1, and `resolution` to `[512, 768, 1024]`.",
        "- **Want the LoRA to inherit the reference outfit?** Generate with",
        "  `vary outfit` OFF; the trigger will pull the reference wardrobe.",
        "- **Want outfit-agnostic?** Generate with `vary outfit` ON (32 outfits",
        "  rotate through the dataset so the LoRA decouples identity from clothes).",
        "",
        "## musubi-tuner (alternate)",
        "",
        "1. Install musubi-tuner: <https://github.com/kohya-ss/musubi-tuner>",
        "2. Docs: <https://github.com/kohya-ss/musubi-tuner/blob/main/docs/qwen_image.md>",
        "3. Cache latents + text-encoder outputs (per musubi docs).",
        "4. Run (replace the `path/to/...` placeholders with your model paths):",
        "",
        "```bash",
        "accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 \\",
        "    src/musubi_tuner/qwen_image_train_network.py \\",
        "    --dit path/to/dit_model \\",
        "    --vae path/to/vae_model \\",
        "    --text_encoder path/to/text_encoder \\",
        "    --model_version original \\",
        '    --dataset_config "_musubi_config.toml" \\',
        "    --sdpa --timestep_sampling shift \\",
        "    --weighting_scheme none --discrete_flow_shift 2.2 \\",
        "    --optimizer_type adamw8bit --learning_rate 5e-5 \\",
        "    --gradient_checkpointing \\",
        "    --max_data_loader_n_workers 2 --persistent_data_loader_workers \\",
        "    --network_module networks.lora_qwen_image \\",
        "    --network_dim 32 \\",
        "    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \\",
        "    --output_dir ./outputs --output_name my_lora",
        "```",
        "",
        "## Caption style",
        "",
        "Captions are single-line natural language, trigger word first. Each `.txt`",
        "sits next to its matching `.png`. This format works for both ai-toolkit and",
        "musubi-tuner out of the box.",
    ]

    (output_dir / "_musubi_config.toml").write_text("\n".join(toml_lines), encoding="utf-8")
    (output_dir / "_training_README.md").write_text("\n".join(readme_lines), encoding="utf-8")


# -------------------------------------------------------------------------
# Review window — per-character thumbnail grid with click-to-reject
# -------------------------------------------------------------------------

class ReviewWindow:
    def __init__(self, parent, char: Character, out_char: Path, api_key: str, model: str):
        self.char = char
        self.out_char = out_char
        self.api_key = api_key
        self.model = model

        self.manifest = load_manifest(out_char)
        self.slots = self._scan_slots()

        self._validation_queue: "queue.Queue" = queue.Queue()
        self._validation_running = False
        self._engine: Optional[GeminiEngine] = None

        self.win = tk.Toplevel(parent)
        self.win.title(f"Review — {char.name}")
        self.win.configure(bg=COLOR["bg"])
        self.win.geometry("1260x920")
        self.win.minsize(1000, 700)
        apply_dark_titlebar(self.win)

        self._thumb_widgets: dict[int, dict] = {}
        self._build_ui()
        self._poll_validation()

    def _scan_slots(self) -> list[dict]:
        out = []
        for img_path in sorted(self.out_char.glob("[0-9][0-9][0-9].png")):
            try:
                slot = int(img_path.stem)
            except ValueError:
                continue
            key = f"{slot:03d}"
            info = self.manifest.get("slots", {}).get(key, {})
            out.append({
                "slot": slot,
                "path": img_path,
                "rejected": not info.get("kept", True),
                "score": info.get("score"),
                "reason": info.get("validate_reason"),
                "flat": info.get("flat"),
            })
        # O(1) index by slot — used by click/poll hot paths that would
        # otherwise scan the full list per event (500 × 500 = 250k cmps
        # on a large validation run).
        self._slot_by_id = {s["slot"]: s for s in out}
        return out

    # -------- UI --------
    def _build_ui(self):
        hdr = tk.Frame(self.win, bg=COLOR["bg"])
        hdr.pack(fill="x", padx=22, pady=(18, 4))
        tk.Label(hdr, text=f"Review — {self.char.name}",
                 bg=COLOR["bg"], fg=COLOR["text"],
                 font=(FONT_UI, 17, "bold")).pack(side="left")
        tk.Label(hdr, text=f"   {len(self.slots)} images on disk",
                 bg=COLOR["bg"], fg=COLOR["muted"],
                 font=(FONT_UI, 10)).pack(side="left", pady=(6, 0))

        tk.Label(
            self.win,
            text=(
                "click any thumbnail to toggle rejection  ·  "
                "validate compares each image to the face reference via Gemini  ·  "
                "on apply, rejected images are deleted and the next Generate pass refills "
                "those slots with fresh (guaranteed-unused) combinatorial specs"
            ),
            bg=COLOR["bg"], fg=COLOR["muted"],
            font=(FONT_UI, 9), wraplength=1180,
            justify="left", anchor="w",
        ).pack(fill="x", padx=22, pady=(0, 10))

        # Scrollable grid
        grid_outer = tk.Frame(self.win, bg=COLOR["surface"], bd=0,
                              highlightthickness=1, highlightbackground=COLOR["border"])
        grid_outer.pack(fill="both", expand=True, padx=22, pady=(0, 10))

        self.canvas = tk.Canvas(grid_outer, bg=COLOR["bg"], highlightthickness=0, bd=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        vs = ttk.Scrollbar(grid_outer, orient="vertical", command=self.canvas.yview,
                           style="Forge.Vertical.TScrollbar")
        vs.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=vs.set)

        self.grid_inner = tk.Frame(self.canvas, bg=COLOR["bg"])
        self._grid_window = self.canvas.create_window((0, 0), window=self.grid_inner, anchor="nw")
        self.canvas.bind("<Configure>",
            lambda e: self.canvas.itemconfigure(self._grid_window, width=e.width))
        self.grid_inner.bind("<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        _bind_wheel_on_hover(self.canvas, self.grid_inner, self.win)

        if not self.slots:
            tk.Label(self.grid_inner, text=f"No generated images found in\n{self.out_char}",
                     bg=COLOR["bg"], fg=COLOR["muted"],
                     font=(FONT_UI, 11, "italic"), pady=80).pack()
        else:
            self._render_grid()

        # Bottom toolbar
        tb = tk.Frame(self.win, bg=COLOR["bg"])
        tb.pack(fill="x", padx=22, pady=(0, 18))

        self.summary_var = tk.StringVar()
        tk.Label(tb, textvariable=self.summary_var, bg=COLOR["bg"],
                 fg=COLOR["text_dim"], font=(FONT_MONO, 10)).pack(side="left")
        self._update_summary()

        HoverButton(tb, "cancel", command=self.win.destroy,
                    variant="ghost", pad=(14, 8)).pack(side="right", padx=(6, 0))
        self.apply_btn = HoverButton(
            tb, "  apply & close  ", command=self._apply_and_close,
            variant="primary", pad=(16, 9), font=(FONT_UI, 10, "bold"),
        )
        self.apply_btn.pack(side="right", padx=6)

        HoverButton(tb, "flag low scores", command=self._flag_low_scores,
                    variant="secondary", pad=(14, 8)).pack(side="right", padx=6)

        self.validate_btn = HoverButton(
            tb, "validate (Gemini judge)", command=self._start_validation,
            variant="secondary", pad=(14, 8),
        )
        self.validate_btn.pack(side="right", padx=6)

    def _render_grid(self):
        cols = REVIEW_COLS
        for c in range(cols):
            self.grid_inner.grid_columnconfigure(c, weight=1, uniform="thumbcol")
        for idx, info in enumerate(self.slots):
            self._render_thumb(info, idx // cols, idx % cols)

    def _render_thumb(self, slot_info, r, c):
        slot = slot_info["slot"]
        frame = tk.Frame(self.grid_inner, bg=COLOR["row"], bd=0,
                         highlightthickness=3, cursor="hand2")
        frame.grid(row=r, column=c, padx=10, pady=10)

        try:
            im = Image.open(slot_info["path"])
            im.thumbnail(THUMB_SIZE, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(im)
        except Exception:
            photo = None

        if photo:
            img_lbl = tk.Label(frame, image=photo, bg=COLOR["row"], bd=0)
            img_lbl.image = photo
            img_lbl.pack(padx=6, pady=(6, 4))
        else:
            img_lbl = tk.Label(frame, text="(image error)", width=20, height=12,
                               bg=COLOR["row"], fg=COLOR["err"], font=(FONT_UI, 9))
            img_lbl.pack(padx=6, pady=6)

        info_row = tk.Frame(frame, bg=COLOR["row"])
        info_row.pack(fill="x", padx=8, pady=(0, 6))
        slot_lbl = tk.Label(info_row, text=f"{slot:03d}", bg=COLOR["row"],
                            fg=COLOR["text"], font=(FONT_MONO, 10, "bold"))
        slot_lbl.pack(side="left")
        score_var = tk.StringVar(value=self._fmt_score(slot_info.get("score")))
        score_lbl = tk.Label(info_row, textvariable=score_var, bg=COLOR["row"],
                             fg=COLOR["text_dim"], font=(FONT_MONO, 9))
        score_lbl.pack(side="right")

        def on_click(_e, s=slot):
            self._toggle_rejection(s)
        for w in (frame, img_lbl, info_row, slot_lbl, score_lbl):
            w.bind("<Button-1>", on_click)

        self._thumb_widgets[slot] = {
            "frame": frame, "img_lbl": img_lbl,
            "score_var": score_var, "score_lbl": score_lbl,
        }
        self._apply_border(slot_info)

    def _fmt_score(self, score):
        if score is None:
            return "—"
        if score >= 0.80:
            return f"★ {score:.2f}"
        if score >= VALIDATE_THRESHOLD:
            return f"  {score:.2f}"
        return f"⚠ {score:.2f}"

    def _apply_border(self, slot_info):
        w = self._thumb_widgets.get(slot_info["slot"])
        if not w:
            return
        if slot_info["rejected"]:
            color = COLOR["err"]
        elif slot_info.get("score") is not None and slot_info["score"] < VALIDATE_THRESHOLD:
            color = COLOR["warn"]
        else:
            color = COLOR["border"]
        w["frame"].configure(highlightbackground=color, highlightcolor=color)

    def _toggle_rejection(self, slot):
        info = self._slot_by_id.get(slot)
        if info is None:
            return
        info["rejected"] = not info["rejected"]
        self._apply_border(info)
        self._update_summary()

    def _flag_low_scores(self):
        n = 0
        for s in self.slots:
            if s.get("score") is not None and s["score"] < VALIDATE_THRESHOLD and not s["rejected"]:
                s["rejected"] = True
                self._apply_border(s)
                n += 1
        self._update_summary()
        if n == 0:
            messagebox.showinfo(
                "no flags",
                "nothing to flag — either no scores exist yet (run Validate first), "
                "or nothing scored below the threshold.",
            )
        else:
            messagebox.showinfo("flagged", f"auto-rejected {n} image(s) scoring below {VALIDATE_THRESHOLD:.2f}")

    def _update_summary(self):
        total = len(self.slots)
        rejected = sum(1 for s in self.slots if s["rejected"])
        scored = sum(1 for s in self.slots if s.get("score") is not None)
        low = sum(1 for s in self.slots
                  if s.get("score") is not None and s["score"] < VALIDATE_THRESHOLD)
        self.summary_var.set(
            f"total {total}  ·  rejected {rejected}  ·  scored {scored}/{total}  "
            f"·  low-score {low}"
        )

    # -------- validation --------
    def _start_validation(self):
        if self._validation_running or not self.slots:
            return
        if not self.api_key:
            messagebox.showerror("no api key", "no API key available for validation")
            return
        try:
            self._engine = GeminiEngine(api_key=self.api_key, model=self.model)
            self._face_bytes = _load_jpeg_bytes(self.char.face)
        except Exception as e:
            messagebox.showerror("init failed", str(e))
            return

        to_score = [s for s in self.slots if s.get("score") is None]
        if not to_score:
            messagebox.showinfo("nothing to do", "every image already has a score.")
            return

        self._validation_running = True
        self.validate_btn.set_enabled(False)

        def worker():
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                futs = {pool.submit(self._validate_one, s): s for s in to_score}
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        slot, score, reason = fut.result()
                        self._validation_queue.put((slot, score, reason))
                    except Exception as e:
                        self._validation_queue.put(("ERR", str(e), None))
            self._validation_queue.put(("DONE", None, None))

        threading.Thread(target=worker, daemon=True).start()

    def _validate_one(self, slot_info):
        with open(slot_info["path"], "rb") as f:
            img_bytes = f.read()
        score, reason = self._engine.validate_face(self._face_bytes, img_bytes)
        return slot_info["slot"], score, reason

    def _poll_validation(self):
        # Bail early if window is gone — no more widget touches.
        if not self.win.winfo_exists():
            return
        try:
            while True:
                slot, score, reason = self._validation_queue.get_nowait()
                try:
                    if slot == "DONE":
                        self._validation_running = False
                        self.validate_btn.set_enabled(True)
                        self._update_summary()
                        continue
                    if slot == "ERR":
                        continue
                    info = self._slot_by_id.get(slot)
                    if info is None:
                        continue
                    info["score"] = score
                    info["reason"] = reason
                    w = self._thumb_widgets.get(slot)
                    if w:
                        w["score_var"].set(self._fmt_score(score))
                        self._apply_border(info)
                    key = f"{slot:03d}"
                    entry = self.manifest.setdefault("slots", {}).setdefault(key, {})
                    entry["score"] = score
                    entry["validate_reason"] = reason
                    save_manifest(self.out_char, self.manifest)
                    self._update_summary()
                except tk.TclError:
                    # Widget destroyed mid-update — stop processing this batch.
                    return
                except StopIteration:
                    continue
        except queue.Empty:
            pass
        try:
            self.win.after(150, self._poll_validation)
        except tk.TclError:
            pass

    # -------- apply --------
    def _apply_and_close(self):
        deleted = 0
        for s in self.slots:
            if s["rejected"]:
                key = f"{s['slot']:03d}"
                entry = self.manifest.setdefault("slots", {}).setdefault(key, {})
                entry["kept"] = False
                if s.get("flat") is not None and entry.get("flat") is None:
                    entry["flat"] = s["flat"]
                try:
                    s["path"].unlink(missing_ok=True)
                    s["path"].with_suffix(".txt").unlink(missing_ok=True)
                    deleted += 1
                except Exception:
                    pass
        save_manifest(self.out_char, self.manifest)
        self.win.destroy()
        if deleted:
            messagebox.showinfo(
                "rejected deleted",
                f"deleted {deleted} image(s). hit GENERATE to refill those slots with fresh specs.",
            )


# -------------------------------------------------------------------------
# Main app
# -------------------------------------------------------------------------

class ForgeApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("LoRA-Dataset-Forge")
        self.root.geometry("1460x900")
        self.root.configure(bg=COLOR["bg"])
        self.root.minsize(1200, 700)

        self.msgq: "queue.Queue[ProgressMsg]" = queue.Queue()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.chars: list[Character] = []
        self.rows: list[CharRow] = []
        self._total_jobs = 0
        self._done_jobs = 0
        self._char_done: dict[str, int] = {}
        self._job_running = False
        self._save_error_logged = False  # suppress repeat save-error spam

        # Auto-save state
        self._settings = load_settings()
        self._save_after_id: Optional[str] = None
        self._settings_ready = False  # don't save while building UI (noise)

        self._setup_fonts()
        self._setup_styles()
        self._build_ui()
        apply_dark_titlebar(self.root)
        self._bind_global_save_traces()
        self._schedule_poll()
        self._scan_and_populate()

        # Enable saving once the initial UI + scan has settled
        self.root.after(600, lambda: setattr(self, "_settings_ready", True))

        # Check for a pending batch from a previous session (after scan + settling)
        self.root.after(800, self._maybe_resume_batch)

        # Save on close
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # -------- setup --------
    def _setup_fonts(self):
        # Warm default font so Entry, Spinbox etc pick it up
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            try:
                tkfont.nametofont(name).configure(family=FONT_UI, size=10)
            except tk.TclError:
                pass

    def _setup_styles(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Combobox
        style.configure("Forge.TCombobox",
                        fieldbackground=COLOR["surface"],
                        background=COLOR["surface2"],
                        foreground=COLOR["text"],
                        arrowcolor=COLOR["text_dim"],
                        bordercolor=COLOR["border"],
                        lightcolor=COLOR["border"],
                        darkcolor=COLOR["border"],
                        selectbackground=COLOR["surface"],
                        selectforeground=COLOR["accent"],
                        padding=(8, 6))
        style.map("Forge.TCombobox",
                  fieldbackground=[("readonly", COLOR["surface"])],
                  foreground=[("readonly", COLOR["text"])])
        self.root.option_add("*TCombobox*Listbox.background", COLOR["surface"])
        self.root.option_add("*TCombobox*Listbox.foreground", COLOR["text"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", COLOR["accent"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#1a0e08")
        self.root.option_add("*TCombobox*Listbox.font", (FONT_UI, 10))

        # Scrollbars (both panels)
        style.configure("Forge.Vertical.TScrollbar",
                        background=COLOR["surface2"],
                        troughcolor=COLOR["bg"],
                        bordercolor=COLOR["bg"],
                        arrowcolor=COLOR["muted"],
                        lightcolor=COLOR["surface2"],
                        darkcolor=COLOR["surface2"],
                        gripcount=0)
        style.map("Forge.Vertical.TScrollbar",
                  background=[("active", "#2d3340"), ("pressed", COLOR["accent_lo"])])

        # Progress bar
        style.configure("Forge.Horizontal.TProgressbar",
                        troughcolor=COLOR["surface"],
                        background=COLOR["accent"],
                        bordercolor=COLOR["bg"],
                        lightcolor=COLOR["accent_hi"],
                        darkcolor=COLOR["accent_lo"],
                        thickness=8)

    # -------- UI build --------
    def _build_ui(self):
        # ---- Header ----
        header = tk.Frame(self.root, bg=COLOR["bg"])
        header.pack(fill="x", padx=28, pady=(22, 6))
        tk.Label(header, text="LoRA-Dataset-Forge", bg=COLOR["bg"], fg=COLOR["text"],
                 font=(FONT_UI, 19, "bold")).pack(side="left")
        tk.Label(header, text="   Nano Banana 2  →  identity-locked datasets for Qwen-Image LoRA",
                 bg=COLOR["bg"], fg=COLOR["muted"],
                 font=(FONT_UI, 10)).pack(side="left", pady=(7, 0))

        # ---- OneDrive warning banner (hidden by default) ----
        self.onedrive_banner = tk.Frame(self.root, bg="#3d2a1a", bd=0,
                                        highlightthickness=1,
                                        highlightbackground=COLOR["warn"])
        tk.Label(
            self.onedrive_banner,
            text=("⚠  dataset root is inside OneDrive — OneDrive can lock files during sync and "
                  "cause PermissionError during generation. If you hit errors, move the dataset "
                  "to a local path like C:/LoRA-Datasets/."),
            bg="#3d2a1a", fg=COLOR["warn"], font=(FONT_UI, 9),
            anchor="w", justify="left", wraplength=1300, padx=16, pady=10,
        ).pack(fill="x")

        # ---- Config card ----
        self.cfg_card = Card(self.root)
        self.cfg_card.pack(fill="x", padx=28, pady=(12, 10))
        cfg = self.cfg_card.inner

        # Row 1: API key + Model
        row1 = tk.Frame(cfg, bg=COLOR["surface"])
        row1.pack(fill="x")
        row1.grid_columnconfigure(1, weight=1)

        s = self._settings  # saved values (may be empty)

        tk.Label(row1, text="API KEY", bg=COLOR["surface"], fg=COLOR["muted"],
                 font=(FONT_UI, 8, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 14))
        # API key preference: saved value wins only if user opted in to remember it.
        self.remember_api_var = tk.BooleanVar(value=bool(s.get("remember_api_key", False)))
        initial_key = os.environ.get("GEMINI_API_KEY", "")
        if self.remember_api_var.get() and s.get("api_key"):
            initial_key = s["api_key"]
        self.api_var = tk.StringVar(value=initial_key)
        api_wrap = tk.Frame(row1, bg=COLOR["surface2"], bd=0, highlightthickness=1,
                            highlightbackground=COLOR["border"])
        api_wrap.grid(row=0, column=1, sticky="ew", padx=(0, 20))
        tk.Entry(api_wrap, textvariable=self.api_var,
                 bg=COLOR["surface2"], fg=COLOR["text"],
                 insertbackground=COLOR["text"], bd=0, relief="flat",
                 show="•", font=(FONT_MONO, 10)).pack(fill="x", padx=12, pady=8)

        # "remember key" toggle sits right after the API key field
        remember_frame = tk.Frame(row1, bg=COLOR["surface"])
        remember_frame.grid(row=0, column=2, sticky="w", padx=(0, 14))
        Toggle(remember_frame, self.remember_api_var, bg=COLOR["surface"]).pack(side="left")
        tk.Label(remember_frame, text=" remember", bg=COLOR["surface"], fg=COLOR["muted"],
                 font=(FONT_UI, 9)).pack(side="left")

        tk.Label(row1, text="MODEL", bg=COLOR["surface"], fg=COLOR["muted"],
                 font=(FONT_UI, 8, "bold")).grid(row=0, column=3, sticky="w", padx=(0, 14))
        self.model_var = tk.StringVar(value=s.get("model", DEFAULT_MODEL))
        cb = ttk.Combobox(row1, textvariable=self.model_var, values=MODEL_CHOICES,
                          width=32, state="readonly", style="Forge.TCombobox")
        cb.grid(row=0, column=4, sticky="w")

        # Row 1b: Resolution + aspect
        row1b = tk.Frame(cfg, bg=COLOR["surface"])
        row1b.pack(fill="x", pady=(14, 0))
        row1b.grid_columnconfigure(5, weight=1)

        tk.Label(row1b, text="RESOLUTION", bg=COLOR["surface"], fg=COLOR["muted"],
                 font=(FONT_UI, 8, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 14))
        self.size_var = tk.StringVar(value=s.get("image_size", DEFAULT_IMAGE_SIZE))
        ttk.Combobox(row1b, textvariable=self.size_var, values=IMAGE_SIZE_CHOICES,
                     width=10, state="readonly", style="Forge.TCombobox").grid(row=0, column=1, sticky="w")

        tk.Label(row1b, text="ASPECT", bg=COLOR["surface"], fg=COLOR["muted"],
                 font=(FONT_UI, 8, "bold")).grid(row=0, column=2, sticky="w", padx=(22, 14))
        self.aspect_var = tk.StringVar(value=s.get("aspect_mode", DEFAULT_ASPECT))
        ttk.Combobox(row1b, textvariable=self.aspect_var, values=ASPECT_CHOICES,
                     width=20, state="readonly", style="Forge.TCombobox").grid(row=0, column=3, sticky="w")

        tk.Label(row1b, text="  2K + smart aspect recommended for Qwen-Image LoRA",
                 bg=COLOR["surface"], fg=COLOR["muted"],
                 font=(FONT_UI, 9, "italic")).grid(row=0, column=4, sticky="w", padx=(24, 0))

        # Row 1bb: generation mode (sync / batch)
        row1bb = tk.Frame(cfg, bg=COLOR["surface"])
        row1bb.pack(fill="x", pady=(14, 0))

        tk.Label(row1bb, text="MODE", bg=COLOR["surface"], fg=COLOR["muted"],
                 font=(FONT_UI, 8, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 14))
        self.mode_var = tk.StringVar(value=s.get("mode", DEFAULT_MODE))

        def _mk_mode_btn(parent, label, value):
            btn = tk.Label(parent, text=label, bg=COLOR["surface2"], fg=COLOR["text_dim"],
                           bd=0, highlightthickness=1, highlightbackground=COLOR["border"],
                           cursor="hand2", padx=14, pady=7, font=(FONT_UI, 10))

            def _refresh(*_):
                on = (self.mode_var.get() == value)
                btn.configure(
                    bg=COLOR["accent"] if on else COLOR["surface2"],
                    fg="#1a0e08" if on else COLOR["text_dim"],
                    highlightbackground=COLOR["accent"] if on else COLOR["border"],
                )
            _refresh()
            self.mode_var.trace_add("write", _refresh)
            btn.bind("<Button-1>", lambda _e, v=value: self.mode_var.set(v))
            return btn

        _mk_mode_btn(row1bb, "sync   (live, full cost)", MODE_SYNC).grid(
            row=0, column=1, sticky="w", padx=(0, 6))
        _mk_mode_btn(row1bb, "batch  (50% off, up to 24h)", MODE_BATCH).grid(
            row=0, column=2, sticky="w", padx=(0, 12))

        self.mode_hint_var = tk.StringVar()
        tk.Label(row1bb, textvariable=self.mode_hint_var,
                 bg=COLOR["surface"], fg=COLOR["muted"],
                 font=(FONT_UI, 9, "italic")).grid(row=0, column=3, sticky="w", padx=(12, 0))

        def _update_mode_hint(*_):
            if self.mode_var.get() == MODE_BATCH:
                self.mode_hint_var.set(
                    "submits an async batch to Google — poll runs in background, results arrive together"
                )
            else:
                self.mode_hint_var.set("images stream back live as each one completes")
        _update_mode_hint()
        self.mode_var.trace_add("write", _update_mode_hint)

        # Row 1c: verify captions + prompt preview + cost estimator
        row1c = tk.Frame(cfg, bg=COLOR["surface"])
        row1c.pack(fill="x", pady=(14, 0))
        row1c.grid_columnconfigure(4, weight=1)

        self.verify_var = tk.BooleanVar(value=bool(s.get("verify_captions", False)))
        verify_frame = tk.Frame(row1c, bg=COLOR["surface"])
        verify_frame.grid(row=0, column=0, sticky="w")
        Toggle(verify_frame, self.verify_var, bg=COLOR["surface"]).pack(side="left")
        tk.Label(verify_frame, text=" verify captions (vision re-caption each image)",
                 bg=COLOR["surface"], fg=COLOR["text_dim"],
                 font=(FONT_UI, 9)).pack(side="left")

        self.preview_btn = HoverButton(row1c, "preview prompt", command=self._open_prompt_preview,
                                       variant="ghost", pad=(14, 7),
                                       font=(FONT_UI, 9))
        self.preview_btn.grid(row=0, column=1, padx=(24, 6))
        self.export_btn = HoverButton(row1c, "export train configs", command=self._export_configs,
                                      variant="ghost", pad=(14, 7),
                                      font=(FONT_UI, 9))
        self.export_btn.grid(row=0, column=2, padx=6)

        # Cost estimator (right-aligned)
        self.cost_var = tk.StringVar(value="est. cost: —")
        tk.Label(row1c, textvariable=self.cost_var,
                 bg=COLOR["surface"], fg=COLOR["accent"],
                 font=(FONT_MONO, 10, "bold")).grid(row=0, column=4, sticky="e", padx=(0, 0))

        # Update estimator whenever any relevant variable changes
        for var in (self.model_var, self.size_var, self.verify_var, self.mode_var):
            try:
                var.trace_add("write", lambda *_: self._update_cost_estimate())
            except tk.TclError:
                pass

        # Row 2: Dataset root
        row2 = tk.Frame(cfg, bg=COLOR["surface"])
        row2.pack(fill="x", pady=(14, 0))
        row2.grid_columnconfigure(1, weight=1)

        tk.Label(row2, text="DATASET", bg=COLOR["surface"], fg=COLOR["muted"],
                 font=(FONT_UI, 8, "bold")).grid(row=0, column=0, sticky="w", padx=(0, 14))
        self.root_var = tk.StringVar(value=s.get("dataset_root", str(DEFAULT_ROOT)))
        root_wrap = tk.Frame(row2, bg=COLOR["surface2"], bd=0, highlightthickness=1,
                             highlightbackground=COLOR["border"])
        root_wrap.grid(row=0, column=1, sticky="ew", padx=(0, 12))
        tk.Entry(root_wrap, textvariable=self.root_var,
                 bg=COLOR["surface2"], fg=COLOR["text_dim"],
                 insertbackground=COLOR["text"], bd=0, relief="flat",
                 font=(FONT_MONO, 10)).pack(fill="x", padx=12, pady=8)

        HoverButton(row2, "browse", command=self._pick_root, variant="secondary",
                    pad=(16, 8)).grid(row=0, column=2, padx=(0, 8))
        HoverButton(row2, "rescan", command=self._scan_and_populate, variant="primary",
                    pad=(16, 8)).grid(row=0, column=3)

        # ---- Table section ----
        tbl_wrap = tk.Frame(self.root, bg=COLOR["bg"])
        tbl_wrap.pack(fill="both", expand=True, padx=28, pady=(8, 8))

        # Section label + counts
        hdr_bar = tk.Frame(tbl_wrap, bg=COLOR["bg"])
        hdr_bar.pack(fill="x", pady=(0, 8))
        tk.Label(hdr_bar, text="CHARACTERS", bg=COLOR["bg"], fg=COLOR["text"],
                 font=(FONT_UI, 10, "bold")).pack(side="left")
        self.count_label = tk.Label(hdr_bar, text="", bg=COLOR["bg"], fg=COLOR["muted"],
                                    font=(FONT_UI, 9))
        self.count_label.pack(side="left", padx=12)

        # Container card
        table_card = tk.Frame(tbl_wrap, bg=COLOR["surface"], bd=0,
                              highlightthickness=1, highlightbackground=COLOR["border"])
        table_card.pack(fill="both", expand=True)

        # Column headers
        self.header_frame = tk.Frame(table_card, bg=COLOR["surface2"])
        self.header_frame.pack(fill="x")
        for col, w in enumerate(COL_WIDTHS):
            self.header_frame.grid_columnconfigure(col, minsize=w, weight=0)

        for col, text in enumerate(["", "CHARACTER", "FACE IMAGE", "BODY IMAGE", "TRIGGER",
                                    "COUNT", "VARIATION", "", "STATUS"]):
            anchor = "e" if col == len(COL_WIDTHS) - 1 else "w"
            tk.Label(self.header_frame, text=text, bg=COLOR["surface2"], fg=COLOR["muted"],
                     font=(FONT_UI, 8, "bold"), anchor=anchor).grid(
                row=0, column=col, sticky="we", padx=COL_PAD_X, pady=12)

        # thin border under header
        tk.Frame(table_card, bg=COLOR["border"], height=1).pack(fill="x")

        # Scrollable list
        list_outer = tk.Frame(table_card, bg=COLOR["surface"])
        list_outer.pack(fill="both", expand=True)

        self.list_canvas = tk.Canvas(list_outer, bg=COLOR["row"], highlightthickness=0, bd=0)
        self.list_canvas.pack(side="left", fill="both", expand=True)

        scroll = ttk.Scrollbar(list_outer, orient="vertical",
                               command=self.list_canvas.yview,
                               style="Forge.Vertical.TScrollbar")
        scroll.pack(side="right", fill="y")
        self.list_canvas.configure(yscrollcommand=scroll.set)

        self.list_inner = tk.Frame(self.list_canvas, bg=COLOR["row"])
        self._list_window = self.list_canvas.create_window((0, 0), window=self.list_inner, anchor="nw")

        def _sync_width(e):
            self.list_canvas.itemconfigure(self._list_window, width=e.width)
        self.list_canvas.bind("<Configure>", _sync_width)
        self.list_inner.bind("<Configure>",
            lambda e: self.list_canvas.configure(scrollregion=self.list_canvas.bbox("all")))

        _bind_wheel_on_hover(self.list_canvas, self.list_inner, self.root)

        # ---- Action toolbar ----
        tb = tk.Frame(self.root, bg=COLOR["bg"])
        tb.pack(fill="x", padx=28, pady=(6, 8))

        HoverButton(tb, "select all", command=lambda: self._bulk_enable(True), variant="secondary",
                    pad=(14, 7)).pack(side="left", padx=(0, 6))
        HoverButton(tb, "deselect all", command=lambda: self._bulk_enable(False), variant="secondary",
                    pad=(14, 7)).pack(side="left", padx=6)

        tk.Label(tb, text="DEFAULT COUNT", bg=COLOR["bg"], fg=COLOR["muted"],
                 font=(FONT_UI, 8, "bold")).pack(side="left", padx=(22, 10))
        self.default_count_var = tk.IntVar(value=int(s.get("default_count", DEFAULT_COUNT)))
        cnt_wrap = tk.Frame(tb, bg=COLOR["surface"], bd=0, highlightthickness=1,
                            highlightbackground=COLOR["border"])
        cnt_wrap.pack(side="left", padx=(0, 6))
        tk.Spinbox(cnt_wrap, from_=1, to=500, textvariable=self.default_count_var, width=6,
                   bg=COLOR["surface"], fg=COLOR["text"],
                   buttonbackground=COLOR["surface2"],
                   bd=0, relief="flat", font=(FONT_MONO, 10),
                   insertbackground=COLOR["text"]).pack(padx=6, pady=5)
        HoverButton(tb, "apply to all", command=self._apply_default_count, variant="secondary",
                    pad=(14, 7)).pack(side="left", padx=6)

        tk.Label(tb, text="WORKERS", bg=COLOR["bg"], fg=COLOR["muted"],
                 font=(FONT_UI, 8, "bold")).pack(side="left", padx=(22, 10))
        self.workers_var = tk.IntVar(value=int(s.get("workers", DEFAULT_WORKERS)))
        wk_wrap = tk.Frame(tb, bg=COLOR["surface"], bd=0, highlightthickness=1,
                           highlightbackground=COLOR["border"])
        wk_wrap.pack(side="left", padx=(0, 6))
        tk.Spinbox(wk_wrap, from_=1, to=MAX_WORKERS_LIMIT, textvariable=self.workers_var,
                   width=4, bg=COLOR["surface"], fg=COLOR["text"],
                   buttonbackground=COLOR["surface2"],
                   bd=0, relief="flat", font=(FONT_MONO, 10),
                   insertbackground=COLOR["text"]).pack(padx=6, pady=5)

        # Action buttons (right)
        self.generate_btn = HoverButton(tb, "  ▶   GENERATE  ", command=self._start_job,
                                        variant="primary", pad=(20, 9),
                                        font=(FONT_UI, 10, "bold"))
        self.generate_btn.pack(side="right", padx=(8, 0))
        self.stop_btn = HoverButton(tb, "  ■  stop  ", command=self._stop_job, variant="secondary",
                                    pad=(16, 9))
        self.stop_btn.pack(side="right", padx=6)
        self.stop_btn.set_enabled(False)

        # ---- Progress strip ----
        prog_wrap = tk.Frame(self.root, bg=COLOR["bg"])
        prog_wrap.pack(fill="x", padx=28, pady=(4, 10))
        self.pbar = ttk.Progressbar(prog_wrap, style="Forge.Horizontal.TProgressbar",
                                    mode="determinate", length=100)
        self.pbar.pack(fill="x", side="left", expand=True)
        self.status_var = tk.StringVar(value="idle")
        tk.Label(prog_wrap, textvariable=self.status_var, bg=COLOR["bg"], fg=COLOR["text_dim"],
                 font=(FONT_MONO, 9), width=24, anchor="e").pack(side="right", padx=(16, 0))

        # ---- Log card ----
        log_wrap = tk.Frame(self.root, bg=COLOR["bg"])
        log_wrap.pack(fill="both", expand=False, padx=28, pady=(0, 22))

        log_hdr = tk.Frame(log_wrap, bg=COLOR["bg"])
        log_hdr.pack(fill="x", pady=(0, 6))
        tk.Label(log_hdr, text="LOG", bg=COLOR["bg"], fg=COLOR["text"],
                 font=(FONT_UI, 10, "bold")).pack(side="left")

        log_card = tk.Frame(log_wrap, bg=COLOR["surface"], bd=0,
                            highlightthickness=1, highlightbackground=COLOR["border"])
        log_card.pack(fill="both", expand=True)
        log_inner = tk.Frame(log_card, bg=COLOR["surface"])
        log_inner.pack(fill="both", expand=True, padx=14, pady=12)
        self.log = tk.Text(
            log_inner, bg=COLOR["surface"], fg=COLOR["text_dim"],
            insertbackground=COLOR["text"], bd=0, relief="flat",
            font=(FONT_MONO, 9), wrap="word", height=9,
        )
        self.log.pack(side="left", fill="both", expand=True)
        log_scroll = ttk.Scrollbar(log_inner, orient="vertical", command=self.log.yview,
                                   style="Forge.Vertical.TScrollbar")
        log_scroll.pack(side="right", fill="y", padx=(8, 0))
        self.log.configure(yscrollcommand=log_scroll.set, state="disabled")
        self.log.tag_config("err",  foreground=COLOR["err"])
        self.log.tag_config("ok",   foreground=COLOR["ok"])
        self.log.tag_config("info", foreground=COLOR["info"])
        self.log.tag_config("dim",  foreground=COLOR["muted"])

    # -------- actions --------
    def _pick_root(self):
        d = filedialog.askdirectory(initialdir=self.root_var.get() or str(DEFAULT_ROOT))
        if d:
            self.root_var.set(d)
            self._scan_and_populate()

    def _scan_and_populate(self):
        if getattr(self, "_job_running", False):
            messagebox.showinfo(
                "busy",
                "cannot rescan while a generation is running. stop first, then rescan.",
            )
            return
        for w in self.list_inner.winfo_children():
            w.destroy()
        self.rows.clear()

        root = Path(self.root_var.get())
        self.chars, scan_warnings = scan_characters(root)
        for w in scan_warnings:
            self._log(f"[scan warn] {w}", "err")

        if not self.chars:
            tk.Label(
                self.list_inner, text=f"no character subfolders found in\n{root}",
                bg=COLOR["row"], fg=COLOR["muted"], justify="center",
                font=(FONT_UI, 10, "italic"), pady=40,
            ).pack(fill="x")
            self.count_label.configure(text="0 characters")
            self._log(f"scan: no characters found under {root}", "dim")
            return

        self.count_label.configure(text=f"{len(self.chars)} character{'s' if len(self.chars)!=1 else ''} detected")
        self._log(f"scan: found {len(self.chars)} character(s) in {root}", "info")

        output_dir = root / DEFAULT_OUTPUT_SUBDIR
        saved_chars = (self._settings.get("characters") or {})
        for i, c in enumerate(self.chars):
            # Restore distribution + count from prior manifest, if any
            char_out = output_dir / c.name
            if char_out.exists():
                m = load_manifest(char_out)
                saved_dist = m.get("distribution")
                if isinstance(saved_dist, dict) and sum(int(v) for v in saved_dist.values() if isinstance(v, int)) > 0:
                    c.distribution = {k: int(v) for k, v in saved_dist.items() if isinstance(v, int)}
                    c.count = sum(c.distribution.values())
            # Overlay persisted UI state (higher priority than manifest)
            saved = saved_chars.get(c.name)
            if isinstance(saved, dict):
                self._apply_saved_char(c, saved)
            self._log(f"  · {c.name}  →  face={c.face.name}, body={c.body.name}, trigger={c.trigger}"
                      + (f", dist={c.distribution}" if c.distribution else ""), "dim")
            row = CharRow(self.list_inner, c, i,
                          self._pick_face_for, self._pick_body_for,
                          self._preview, self._open_review, self._edit_distribution,
                          on_any_change=self._schedule_save)
            row.pack(fill="x")
            self.rows.append(row)

        self._wire_row_cost_traces()
        self._update_cost_estimate()
        self._check_onedrive()

    def _pick_face_for(self, row: CharRow):
        p = filedialog.askopenfilename(initialdir=row.char.folder,
                                       filetypes=[("images", "*.jpg *.jpeg *.png *.webp")])
        if p:
            row.face_picker.set_path(Path(p))

    def _pick_body_for(self, row: CharRow):
        p = filedialog.askopenfilename(initialdir=row.char.folder,
                                       filetypes=[("images", "*.jpg *.jpeg *.png *.webp")])
        if p:
            row.body_picker.set_path(Path(p))

    def _preview(self, row: CharRow):
        # Dedup — if a preview window is already open for this char, lift it
        # rather than opening another one (each holds ~20MB of ref thumbnails).
        if not hasattr(self, "_preview_windows"):
            self._preview_windows: dict[str, tk.Toplevel] = {}
        existing = self._preview_windows.get(row.char.name)
        if existing and existing.winfo_exists():
            existing.lift()
            existing.focus_set()
            return

        win = tk.Toplevel(self.root)
        win.title(f"{row.char.name}  —  references")
        win.configure(bg=COLOR["bg"])
        apply_dark_titlebar(win)
        self._preview_windows[row.char.name] = win
        win.protocol("WM_DELETE_WINDOW",
                     lambda n=row.char.name: self._close_preview(n))

        for label, path in [("face", row.face_picker.path), ("body", row.body_picker.path)]:
            col = tk.Frame(win, bg=COLOR["bg"])
            col.pack(side="left", padx=18, pady=18)
            tk.Label(col, text=label.upper(), bg=COLOR["bg"], fg=COLOR["muted"],
                     font=(FONT_UI, 9, "bold")).pack(anchor="w", pady=(0, 2))
            tk.Label(col, text=path.name, bg=COLOR["bg"], fg=COLOR["text_dim"],
                     font=(FONT_MONO, 9)).pack(anchor="w", pady=(0, 10))
            try:
                im = Image.open(path)
                im.thumbnail((460, 640))
                photo = ImageTk.PhotoImage(im)
                lbl = tk.Label(col, image=photo, bg=COLOR["bg"], bd=0)
                lbl.image = photo
                lbl.pack()
            except Exception as e:
                tk.Label(col, text=f"(err: {e})", bg=COLOR["bg"], fg=COLOR["err"]).pack()

    def _close_preview(self, char_name: str):
        win = self._preview_windows.pop(char_name, None)
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass

    # -------- auto-save --------
    def _bind_global_save_traces(self):
        for var in (self.api_var, self.model_var, self.size_var, self.aspect_var,
                    self.default_count_var, self.workers_var, self.verify_var,
                    self.root_var, self.remember_api_var, self.mode_var):
            try:
                var.trace_add("write", lambda *_: self._schedule_save())
            except tk.TclError:
                pass

    def _schedule_save(self):
        if not getattr(self, "_settings_ready", False):
            return
        if self._save_after_id:
            try:
                self.root.after_cancel(self._save_after_id)
            except tk.TclError:
                pass
        self._save_after_id = self.root.after(SETTINGS_DEBOUNCE_MS, self._do_save)

    def _do_save(self):
        self._save_after_id = None
        try:
            save_settings(self._collect_settings())
            # reset the error-log suppression flag on first success after failure
            self._save_error_logged = False
        except Exception as e:
            if not self._save_error_logged:
                try:
                    self._log(f"[save] settings save failed — changes this session may be lost: {e}", "err")
                except Exception:
                    pass
                self._save_error_logged = True

    def _collect_settings(self) -> dict:
        chars_data: dict[str, dict] = {}
        for r in self.rows:
            try:
                chars_data[r.char.name] = {
                    "enabled": bool(r.enabled_var.get()),
                    "face": str(r.face_picker.path),
                    "body": str(r.body_picker.path),
                    "trigger": r.trigger_var.get().strip() or r.char.trigger,
                    "count": int(r.count_var.get()),
                    "vary_outfit": bool(r.vary_var.get()),
                    "distribution": r.char.distribution,
                }
            except (tk.TclError, ValueError):
                continue
        remember_key = bool(self.remember_api_var.get())
        return {
            "version": SETTINGS_VERSION,
            # API key is persisted only when the "remember" toggle is on.
            # Otherwise it stays in memory for this session only.
            "remember_api_key": remember_key,
            "api_key": self.api_var.get() if remember_key else "",
            "model": self.model_var.get(),
            "image_size": self.size_var.get(),
            "aspect_mode": self.aspect_var.get(),
            "default_count": int(self.default_count_var.get()),
            "workers": int(self.workers_var.get()),
            "verify_captions": bool(self.verify_var.get()),
            "dataset_root": self.root_var.get(),
            "mode": self.mode_var.get(),
            "characters": chars_data,
        }

    def _apply_saved_char(self, char: Character, saved: dict):
        """Overlay saved per-character settings onto a freshly-scanned Character."""
        if "enabled" in saved:
            char.enabled = bool(saved["enabled"])
        if "trigger" in saved and saved["trigger"]:
            char.trigger = str(saved["trigger"])
        if "count" in saved:
            try:
                char.count = int(saved["count"])
            except (TypeError, ValueError):
                pass
        if "vary_outfit" in saved:
            char.vary_outfit = bool(saved["vary_outfit"])
        for key in ("face", "body"):
            raw = saved.get(key)
            if raw:
                p = Path(raw)
                if p.exists():
                    setattr(char, key, p)
        dist = saved.get("distribution")
        if isinstance(dist, dict):
            clean = {k: int(v) for k, v in dist.items()
                     if k in P.CATEGORY_ORDER and isinstance(v, (int, float))}
            if sum(clean.values()) > 0:
                char.distribution = clean
                char.count = sum(clean.values())

    def _on_close(self):
        # Signal any running worker thread to halt — a sync job's
        # ThreadPoolExecutor contains non-daemon threads, and without this
        # signal they'll hold the interpreter open until their 300s Gemini
        # timeout drains. Batch poll threads are daemons so they exit with
        # the process; setting the event just short-circuits any in-flight
        # polling loop immediately.
        try:
            self.stop_event.set()
        except Exception:
            pass
        # Flush pending save synchronously
        if self._save_after_id:
            try:
                self.root.after_cancel(self._save_after_id)
            except tk.TclError:
                pass
        try:
            save_settings(self._collect_settings())
        except Exception:
            pass
        self.root.destroy()

    # -------- cost estimator --------
    def _update_cost_estimate(self):
        if not getattr(self, "rows", None) or not hasattr(self, "cost_var"):
            return
        total = 0
        for r in self.rows:
            try:
                if r.enabled_var.get():
                    total += int(r.count_var.get())
            except (tk.TclError, ValueError):
                pass
        if total == 0:
            self.cost_var.set("est. cost: — (no characters enabled)")
            return
        mode = self.mode_var.get() if hasattr(self, "mode_var") else MODE_SYNC
        dollars = estimate_cost(total, self.model_var.get(), self.size_var.get(),
                                verify_captions=bool(self.verify_var.get()),
                                mode=mode)
        verify_tag = " + verify" if self.verify_var.get() else ""
        mode_tag = " · batch 50% off" if mode == MODE_BATCH else ""
        self.cost_var.set(f"est. cost: ${dollars:,.2f}  ({total} images{verify_tag}){mode_tag}")

    def _wire_row_cost_traces(self):
        """After scanning, hook each row's count/enabled vars to cost updater."""
        for r in self.rows:
            for var in (r.count_var, r.enabled_var):
                try:
                    var.trace_add("write", lambda *_: self._update_cost_estimate())
                except tk.TclError:
                    pass

    # -------- onedrive banner --------
    def _check_onedrive(self):
        root_str = str(self.root_var.get()).lower()
        in_onedrive = "onedrive" in root_str
        if in_onedrive and not self.onedrive_banner.winfo_ismapped():
            self.onedrive_banner.pack(fill="x", padx=28, pady=(0, 6),
                                      before=self.cfg_card)
        elif not in_onedrive and self.onedrive_banner.winfo_ismapped():
            self.onedrive_banner.pack_forget()

    # -------- prompt preview --------
    def _open_prompt_preview(self):
        if self._job_running:
            messagebox.showinfo("busy", "cannot open prompt preview while generation is running. stop first.")
            return
        # Pick first enabled char, or first if none enabled
        chars_snap = [r.apply_back() for r in self.rows]
        target = next((c for c in chars_snap if c.enabled), chars_snap[0] if chars_snap else None)
        if not target:
            messagebox.showinfo("no characters", "scan a dataset first")
            return
        PromptPreviewWindow(self.root, target)

    # -------- trainer config export --------
    def _export_configs(self):
        if self._job_running:
            messagebox.showinfo("busy", "cannot export while generation is running. stop first.")
            return
        chars_snap = [r.apply_back() for r in self.rows if r.enabled_var.get()]
        if not chars_snap:
            messagebox.showwarning("nothing to export", "enable at least one character first")
            return
        root = Path(self.root_var.get())
        output_dir = root / DEFAULT_OUTPUT_SUBDIR
        try:
            export_trainer_configs(chars_snap, output_dir)
        except Exception as e:
            messagebox.showerror("export failed", str(e))
            return
        messagebox.showinfo(
            "exported",
            f"wrote:\n"
            f" · _musubi_config.toml\n"
            f" · _aitoolkit_config.yaml\n"
            f" · _training_README.md\n\nto:\n{output_dir}",
        )
        self._log(f"exported train configs → {output_dir}", "ok")

    def _edit_distribution(self, row: "CharRow"):
        if self._job_running:
            messagebox.showinfo("busy", "cannot edit distribution while generation is running. stop first.")
            return
        DistributionEditor(self.root, row)
        self._update_cost_estimate()

    def _open_review(self, row: "CharRow"):
        if self._job_running:
            messagebox.showinfo("busy", "cannot open review while generation is running. stop first.")
            return
        # Snapshot row state back into the character so ReviewWindow has current paths/trigger
        row.apply_back()
        root = Path(self.root_var.get())
        out_char = root / DEFAULT_OUTPUT_SUBDIR / row.char.name
        if not out_char.exists():
            messagebox.showinfo(
                "nothing to review",
                f"no generated images yet for {row.char.name}.\n\n"
                f"expected at: {out_char}\n\nhit GENERATE first.",
            )
            return
        ReviewWindow(self.root, row.char, out_char,
                     api_key=self.api_var.get().strip(),
                     model=self.model_var.get())

    def _bulk_enable(self, val: bool):
        for r in self.rows:
            r.enabled_var.set(val)

    def _apply_default_count(self):
        n = int(self.default_count_var.get())
        for r in self.rows:
            r.count_var.set(n)

    def _start_job(self):
        if self._job_running:
            messagebox.showinfo("busy",
                "a job is already running — stop it first, or wait for the current batch to finish.")
            return
        api_key = self.api_var.get().strip()
        if not api_key:
            messagebox.showerror("no api key", "enter your GEMINI_API_KEY")
            return
        # Deep-copy a snapshot so the worker thread sees an immutable view —
        # the user can freely edit UI state (including distribution, count,
        # trigger) during generation without racing the worker.
        live_chars = [r.apply_back() for r in self.rows]
        chars = [copy.deepcopy(c) for c in live_chars]
        enabled = [c for c in chars if c.enabled]
        if not enabled:
            messagebox.showwarning("nothing to do", "no characters enabled")
            return

        # Validate each enabled character before burning any budget.
        bad = []
        trigger_to_names: dict[str, list[str]] = {}
        for c in enabled:
            if not c.trigger or not c.trigger.strip():
                bad.append(f"{c.name}: empty trigger word")
                continue
            effective_count = sum(c.distribution.values()) if c.distribution else c.count
            if effective_count <= 0:
                bad.append(f"{c.name}: count must be ≥ 1 (got {effective_count})")
            trigger_to_names.setdefault(c.trigger.strip(), []).append(c.name)
        # Detect trigger collisions — same trigger ⇒ identical combinatorial seed
        collisions = [(t, ns) for t, ns in trigger_to_names.items() if len(ns) > 1]
        for t, ns in collisions:
            bad.append(f"trigger {t!r} shared by multiple characters ({', '.join(ns)}) — they would generate IDENTICAL datasets. Make each trigger unique.")
        if bad:
            messagebox.showerror("cannot start", "Fix these first:\n\n - " + "\n - ".join(bad))
            # Note: _job_running is still False here (guard at the top of this
            # method exits early if it was True), so no state to reset.
            return

        root = Path(self.root_var.get())
        output_dir = root / DEFAULT_OUTPUT_SUBDIR
        output_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"output → {output_dir}", "info")

        total = sum(c.count for c in enabled)
        self.pbar.configure(maximum=total, value=0)
        self._total_jobs = total
        self._done_jobs = 0
        self._char_done = {c.name: 0 for c in enabled}
        self.status_var.set(f"0 / {total}")

        for r in self.rows:
            r.status.set_state("queued" if r.enabled_var.get() else "idle")

        self.stop_event.clear()
        self._job_running = True
        self.generate_btn.set_enabled(False)
        self.stop_btn.set_enabled(True)

        mode = self.mode_var.get()
        if mode == MODE_BATCH:
            self._log("submitting batch to Google (will poll in background)...", "info")
            self.worker = threading.Thread(
                target=run_batch_submit,
                args=(chars, output_dir, api_key, self.model_var.get(),
                      self.size_var.get(), self.aspect_var.get(),
                      bool(self.verify_var.get()),
                      self.msgq, self.stop_event),
                daemon=True,
            )
        else:
            self.worker = threading.Thread(
                target=run_job,
                args=(chars, root, output_dir, api_key, self.model_var.get(),
                      self.size_var.get(), self.aspect_var.get(),
                      int(self.workers_var.get()),
                      bool(self.verify_var.get()),
                      self.msgq, self.stop_event),
                daemon=True,
            )
        self.worker.start()

    def _stop_job(self):
        self._log("stop requested — finishing current image then halting…", "dim")
        self.stop_event.set()

    def _spawn_batch_poll(self, output_dir: Path):
        """Start the background poller for a submitted batch."""
        api_key = self.api_var.get().strip()
        if not api_key:
            self._log("[batch] no API key; cannot poll", "err")
            return
        # Wait briefly for the prior (submit) thread to finish emitting its
        # final messages before clobbering self.worker. The submit thread
        # emits batch_submitted immediately before returning, so this join
        # normally completes in <1ms; the timeout is defense-in-depth.
        if self.worker is not None and self.worker.is_alive():
            try:
                self.worker.join(timeout=2.0)
            except Exception:
                pass
        # Don't clear stop_event here — if a submit thread is still unwinding
        # after a Stop press, clearing would silently re-enable it. The submit
        # path clears stop_event in _start_job when the user clicks Generate.
        self._job_running = True
        self.generate_btn.set_enabled(False)
        self.stop_btn.set_enabled(True)
        self.worker = threading.Thread(
            target=run_batch_poll,
            args=(output_dir, api_key, self.model_var.get(),
                  self.msgq, self.stop_event),
            daemon=True,
        )
        self.worker.start()

    def _maybe_resume_batch(self):
        """On launch, check for a pending batch state file and offer to resume polling."""
        root = Path(self.root_var.get())
        output_dir = root / DEFAULT_OUTPUT_SUBDIR
        state = load_batch_state(output_dir)
        if not state:
            return
        name = state.get("batch_name", "?")
        total = state.get("total_requests", 0)

        # Cross-root sanity: if the state file references characters that don't
        # exist under the current dataset root, user likely changed datasets
        # between sessions. Warn instead of silently writing to wrong folders.
        char_names_in_state = {
            info.get("char") for info in (state.get("key_map") or {}).values()
        }
        char_names_in_state.discard(None)
        scanned_names = {c.name for c in self.chars}
        missing = char_names_in_state - scanned_names
        if missing and char_names_in_state:
            if not messagebox.askyesno(
                "batch references different characters",
                f"The pending batch was submitted for character(s) that don't "
                f"exist under the current dataset root:\n\n"
                f"  missing: {', '.join(sorted(missing))}\n"
                f"  current: {', '.join(sorted(scanned_names)) or '(none)'}\n\n"
                f"Did you change the dataset root since submitting?\n\n"
                f"Resume anyway? (Choose No to leave the batch alone; you can "
                f"point the dataset root back at the original folder and "
                f"relaunch to resume safely.)",
            ):
                self._log("[batch] resume declined — pending state preserved", "info")
                return

        self._log(f"[batch] pending batch detected: {name} ({total} req)", "info")

        # Pre-populate progress state so the status bar shows sensible counts
        # when results start arriving, instead of "N/0" garbage.
        try:
            t = int(total)
        except (ValueError, TypeError):
            t = 0
        if t > 0:
            self._total_jobs = t
            self._done_jobs = 0
            self._char_done = {}
            try:
                self.pbar.configure(maximum=t, value=0)
            except tk.TclError:
                pass

        if not messagebox.askyesno(
            "pending batch",
            f"a submitted batch is still waiting for results:\n\n"
            f"  {name}\n  {total} request(s)\n\n"
            f"resume polling now?",
        ):
            return

        # If no API key present, prompt the user to enter one and retry.
        if not self.api_var.get().strip():
            messagebox.showwarning(
                "api key needed",
                "enter your GEMINI_API_KEY in the config card, then "
                "click GENERATE to trigger a resume (the tool will detect "
                "the pending batch again).",
            )
            return

        self._spawn_batch_poll(output_dir)

    # -------- log + polling --------
    _LOG_MAX_LINES = 2000
    _LOG_TRIM_TO = 1500

    def _log(self, text: str, tag: str = ""):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag)
        # Keep the log bounded so a long session doesn't grow memory forever.
        try:
            lines = int(self.log.index("end-1c").split(".")[0])
            if lines > self._LOG_MAX_LINES:
                self.log.delete("1.0", f"{lines - self._LOG_TRIM_TO}.0")
        except (tk.TclError, ValueError):
            pass
        self.log.see("end")
        self.log.configure(state="disabled")

    def _schedule_poll(self):
        try:
            while True:
                msg = self.msgq.get_nowait()
                self._handle_msg(msg)
        except queue.Empty:
            pass
        try:
            self.root.after(120, self._schedule_poll)
        except tk.TclError:
            pass  # root destroyed

    def _handle_msg(self, msg: ProgressMsg):
        if msg.kind == "batch_submitted":
            # Re-align progress tracking with the actual batch size (which may
            # be smaller than sum(char.count) if some slots were already present).
            try:
                total_in_batch = int(msg.current)
            except (ValueError, TypeError):
                total_in_batch = 0
            if total_in_batch > 0:
                self._total_jobs = total_in_batch
                self._done_jobs = 0
                self._char_done = {}
                try:
                    self.pbar.configure(maximum=total_in_batch, value=0)
                except tk.TclError:
                    pass
            self._log("batch accepted — polling for results (Ctrl+stop to pause poll)", "ok")
            self.status_var.set(f"batch submitted · 0/{total_in_batch}")
            root = Path(self.root_var.get())
            output_dir = root / DEFAULT_OUTPUT_SUBDIR
            self._spawn_batch_poll(output_dir)
            return
        if msg.kind == "batch_status":
            self.status_var.set(f"batch state: {msg.text}")
            return
        if msg.kind == "log":
            self._log(msg.text)
        elif msg.kind == "error":
            self._log("ERROR: " + msg.text, "err")
            if msg.char:
                for r in self.rows:
                    if r.char.name == msg.char:
                        r.status.set_state("error")
        elif msg.kind == "progress":
            self._done_jobs += 1
            self.pbar.configure(value=self._done_jobs)
            self.status_var.set(f"{self._done_jobs} / {self._total_jobs}")
            done_here = self._char_done.get(msg.char, 0) + 1
            self._char_done[msg.char] = done_here
            for r in self.rows:
                if r.char.name == msg.char:
                    r.status.set_state("running", f"{done_here}/{msg.total}")
        elif msg.kind == "char_start":
            self._char_done[msg.char] = 0
            for r in self.rows:
                if r.char.name == msg.char:
                    r.status.set_state("running", f"0/{msg.total}")
            self._log(f"==> {msg.char}", "ok")
        elif msg.kind == "char_done":
            for r in self.rows:
                if r.char.name == msg.char:
                    r.status.set_state("done")
            self._log(f"<== {msg.char}", "ok")
        elif msg.kind == "done":
            self._job_running = False
            self.generate_btn.set_enabled(True)
            self.stop_btn.set_enabled(False)
            self.status_var.set("idle")
            self._log("== batch finished ==", "ok")


def main():
    root = tk.Tk()
    try:
        root.tk.call("tk", "scaling", 1.15)
    except tk.TclError:
        pass
    ForgeApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
