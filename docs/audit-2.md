# LoRA-Dataset-Forge — Second Audit (post-batch-mode)

Date: 2026-04-18
Scope: full codebase with focus on new batch/Files API code path.
Prior audit: `audit.md` — verify no regressions + find new bugs.

## Summary

Codebase is in good shape. Previous P0/P1 fixes all held. The new batch mode introduced ~600 lines; I found **2 real money-safety risks**, **3 robustness bugs**, and **a handful of UX/polish issues**. None are catastrophic — the wait-for-ACTIVE fix already eliminated the biggest money burn.

## P0 — could bill or corrupt data

### 1. Large batch result download loads entire JSONL into memory
`run_batch_poll` downloads `result_bytes`, decodes to `text`, then `text.splitlines()` — everything resident at once. For a 240-image batch at 2K, each image is ~2MB of base64 JSON → ~480MB in memory. For a 500-image batch, ~1GB. On a machine with 8GB RAM and other apps open, this can swap hard or OOM.
**Fix:** download to a temp file, then stream-iterate by line. Process + save each response then discard.

### 2. Progress bar total not set for batch mode
`_start_job` sets `pbar.configure(maximum=sum(c.count))` assuming all slots will be filled. Batch mode only submits MISSING slots. If 10 slots already exist and count=30, batch is 20 requests but pbar max=30 — progress fills to 20/30 and stops. Visual "job stuck" at 66%.
**Fix:** update `pbar.configure(maximum=...)` after `build_batch_jsonl` returns, using `len(key_map)` as the real total. Also update `_total_jobs` and reset `_char_done`.

## P1 — robustness

### 3. Resume-batch requires API key in the UI at relaunch time
If "remember API key" is OFF and `GEMINI_API_KEY` env var isn't set, `_spawn_batch_poll` refuses with `[batch] no API key; cannot poll`. The user has to manually re-enter the key then re-trigger resume — but there's no "retry resume" button, so they'd have to delete `_batch.json` and start over (losing the pending Google-side batch).
**Fix:** in `_maybe_resume_batch`, if no API key is present, show a message explaining the situation and offer "enter key now and resume" as a dialog action.

### 4. Upload refs leak if `build_batch_jsonl` fails mid-character
Refs for char A upload successfully → char B's face upload fails → we `continue` to char C → char C uploads successfully → later step (e.g., JSONL write) raises → build returns partial work, caller doesn't cleanup. Leaked refs auto-expire at 48h. Not a money burn (storage is free) but messy.
**Fix:** wrap `build_batch_jsonl` in a try/except that deletes any already-uploaded `uploaded_names` on failure.

### 5. Resume after batch SUCCEEDED but pre-cleanup crash → re-download overwrite
If app crashes between `download` and `clear_batch_state`, `_batch.json` persists pointing to a succeeded batch. On resume, results download again and overwrite existing files (same content, but extra bandwidth + overwriting manifest entries for slots that had e.g. validated scores).
**Fix:** check file existence before overwriting on batch result processing, skip if already present. Preserves review-validation scores from an earlier session.

## P2 — polish

### 6. `progress` msg in `run_batch_poll` uses `total=0`
`_handle_msg` renders row status as `done/0`. Cosmetic.
**Fix:** use the per-char slot count.

### 7. HTTP timeout is 300s — probably fine but untested for large result downloads
A 1GB download over a slow connection could exceed 300s. Hasn't been seen in practice.
**Fix:** leave alone; monitor.

### 8. Wait-for-ACTIVE is sequential — 14 uploads × ~8s = ~2 min pre-submit delay for a 7-char batch
Could parallelize uploads but adds complexity.
**Fix:** leave alone for now. Monitor if users complain.

### 9. `_handle_msg` "== batch finished ==" is ambiguous between sync and batch
Both modes emit `done`. The log line reads OK for both but is a double meaning.
**Fix:** differentiate tags or leave alone.

## Regression check against prior audit

Walked through the 9 P0s and 6 P1s from `audit.md`:

| ID | Prior fix | Status |
|---|---|---|
| 1 | `plan_indices` robust sampling | ✅ held |
| 2 | `scan_characters` permission tolerance | ✅ held |
| 3 | `_poll_validation` TclError safety | ✅ held |
| 4 | deep-copy chars at `_start_job` | ✅ held; applied to both sync + batch paths |
| 5 | `DistributionEditor._clear` fires save | ✅ held |
| 6 | `_edit_distribution` no apply_back | ✅ held |
| 7 | scoped mousewheel | ✅ held |
| 8 | interlock (review/dist/preview/export disabled during job) | ✅ held |
| 9 | engine bytes via args not instance state | ✅ held |
| 10 | manifest-category preferred on refill | ✅ held; also used in batch path |
| 11 | tmp file cleanup on atomic write failure | ✅ held; batch state file also uses this pattern |
| 12 | FilePicker.set_path public | ✅ held |
| 13 | surface save failures | ✅ held |
| 14 | log truncation | ✅ held |

No regressions. All prior fixes still in effect and extended to the new batch code path where applicable.
