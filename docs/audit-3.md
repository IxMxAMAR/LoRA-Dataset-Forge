# LoRA-Dataset-Forge — Third Audit (deep pass)

Date: 2026-04-18
Scope: full codebase, three independent reviewer focuses + self pass.
Prior audits: audit.md, audit-2.md — regression baseline + continued exploration.

## Reviewers dispatched

1. Security / data integrity / file I/O
2. UI widgets / threading / lifecycle
3. Prompt quality / export correctness / README accuracy

Plus self pass: pyflakes static analysis, prompt pool duplicate/coverage checks, README-vs-code cross-reference.

## P0 — confirmed real bugs

### 1. **Training command exports a nonexistent module** — users who copy-paste can't train
**Files:** `forge.py` in `export_trainer_configs`, `README.md`

The exported `_training_README.md` prescribes:
```bash
accelerate launch qwen_image_train_network.py \
    --network_module networks.lora_qwen_image \
    ...
```
Both the script name and the module are fabricated — musubi-tuner's actual Qwen-Image trainer uses `networks.lora` with `qwen2_vl_train_network.py` (or the newer dedicated entrypoint). A user who runs the exported command hits `ModuleNotFoundError` immediately. This is the worst kind of bug because the tool ADVERTISES this command and it fails at the finish line.

**Fix:** replace with a command that actually works, OR mark the block as "verify against your musubi-tuner build" with a link.

### 2. **Path traversal via unsanitized `char.name`**
**Files:** `forge.py:scan_characters`, every `output_dir / char.name` use

`char.name` derives from a filesystem subfolder name with only ` retouch` stripped. A malicious / accidental subfolder named `../../../Windows/Temp` would resolve outside the dataset root — `mkdir(parents=True, exist_ok=True)` creates arbitrary directories, and `save_manifest` writes JSON there.

**Reproduction:** place a subfolder named `..\exfil` under the dataset root, scan, generate.

**Fix:** resolve the computed output path and assert it stays under `output_dir`. Sanitize `clean` to strip leading dots / separators. Skip with a warning if traversal detected.

### 3. **README pipeline diagram claims 48 framings** — actual count is 10
**File:** `README.md` pipeline block

```
48 framings × 10 angles × 14 expressions × 18 lightings × 52 scenes ...
```

Actual code: `len(FRAMINGS) == 10`. The "48" is pure fabrication. Misleads contributors about the axis design.

**Fix:** correct to `10 framings` and recompute the headline combinatorial total (it's ~4 billion with outfits varied; the math was right — only the first number is wrong).

## P1 — real robustness issues

### 4. **`_scan_and_populate` is not gated by `_job_running`**
**File:** `forge.py:rescan` button wiring

Every other destructive action (review, dist editor, prompt preview, export) checks `_job_running`. Rescan doesn't — user can click "rescan" while a sync job is live. Rows get destroyed + rebuilt. `_handle_msg`'s `for r in self.rows` iterations then silently find nothing; status pills freeze, `_char_done` drifts.

**Fix:** add the same guard. Show "busy" message.

### 5. **Tk var traces on `Toggle` widgets leak** after row destroy
**File:** `forge.py:Toggle.__init__`

`variable.trace_add("write", lambda *_: self._redraw())` — the lambda captures the Toggle (dead Label after rescan). If the BooleanVar outlives the Toggle, `_redraw()` fires into a destroyed widget → TclError. CPython refcounting cleans up fast enough in normal cases; the hazard is real on rescan + deferred GC.

Also applies to `CharRow._fire_change` traces on enabled_var / trigger_var / count_var / vary_var — captured `self` references keep old CharRows alive after rescan, and their `_fire_change → _schedule_save` path iterates the NEW `self.rows`.

**Fix:** track trace ids, override `destroy()` to call `trace_remove`.

### 6. **Name collision on NTFS (case-insensitive FS)**
**File:** `forge.py:scan_characters` + all `output_dir / char.name` uses

Two source folders `Alice/` and `alice/` pass scan as two distinct Character objects. `output_dir / char.name` resolves to the same dir → both chars write the same manifest, corrupting each other.

**Fix:** post-scan, dedupe by case-folded name; warn and skip duplicates.

### 7. **Download still fully buffered before temp write** — audit-2 fix was half
**File:** `forge.py:run_batch_poll`

`result_bytes = engine.client.files.download(...)` returns all bytes in memory. `tmpf.write(result_bytes)` → now also on disk. `del result_bytes` frees late. Peak RAM is still ~1GB for a 500-image batch.

**Fix:** the SDK doesn't expose a streaming download API in current versions. Document as known limit; user should keep batches ≤ 250 images on memory-constrained machines.

### 8. **ai-toolkit export uses `resolution: [1024]`** — field name and shape may be wrong
**File:** `forge.py:export_trainer_configs`

ai-toolkit dataset YAML accepts `resolution: 1024` (scalar) or a bucket dict. `[1024]` as a single-element list is ambiguous — PyYAML might coerce, ai-toolkit might reject.

**Fix:** use scalar `resolution: 1024` to match documented form.

### 9. **Caption double-trigger** may dilute LoRA identity binding
**File:** `prompts.py:build_caption`

```
f"{trigger}, {spec['framing']} of {trigger}{outfit_phrase}, ..."
```

Trigger appears twice — once at position 0 as the LoRA anchor, again embedded as "framing of ck_xxx". Kohya/musubi convention uses the trigger once, at the start. The second occurrence re-associates the trigger with the framing descriptor rather than the identity.

**Fix:** drop the second trigger. Result: `"ck_chantalle, close-up portrait, ..."` instead of `"ck_chantalle, close-up portrait of ck_chantalle, ..."`.

### 10. **`_preview` windows leak memory** when opened repeatedly
**File:** `forge.py:ForgeApp._preview`

Clicking "preview" on a row opens a Toplevel with two full-res PhotoImages. No `grab_set`, no de-duplication — user can open N concurrent preview windows, each holding ~20MB of image data.

**Fix:** track per-row preview Toplevel; on reopen, `lift()` the existing one instead of creating a new instance.

### 11. **Degenerate outfit + scene combinations** produce nonsense training data
**File:** `prompts.py` (structural)

Pure Cartesian product yields pairs like `activewear + bar + wine glass`, `trench coat + yoga mat`, `chunky sweater + bike shorts + snow`. Nothing explicit, but Gemini either refuses or produces training-useless images.

**Fix:** add a single line to `SYSTEM_PROMPT`: "If the outfit and environment would be contextually incoherent, adapt the environment to suit the outfit." Low-cost, high-value.

## P2 — polish

### 12. `bind_wheel_on_hover` still hijacks globally via `bind_all` — when two Toplevels are open, the last `_leave` kills scroll in both
### 13. `ReviewWindow._render_grid` synchronous at 500 thumbnails → 3-10s freeze on open
### 14. Unused imports: `dataclasses.field` (pyflakes)
### 15. `total` assigned but never used in `_aspect_table` (pyflakes line 2056)
### 16. Trailing f-strings without placeholders (pyflakes ~7 cases — cosmetic)
### 17. Mode button trace leaks on startup (registered on mode_var once per render, never removed)
### 18. Near-duplicate LIGHTINGS entries reduce effective diversity slightly

## Clean items confirmed

- All pools (FRAMINGS, ANGLES, EXPRESSIONS, LIGHTINGS, OUTFITS, WEATHERS, PROPS non-None, SCENES) have zero duplicates
- No TODO/FIXME/XXX/HACK markers
- All prior P0/P1 fixes from audit.md and audit-2.md held through this review
- 14 classes, 122 functions, ~2900 lines forge.py + ~580 lines prompts.py
