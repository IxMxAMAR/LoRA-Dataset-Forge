# LoRA-Dataset-Forge — Audit

Date: 2026-04-18
Scope: full codebase review (forge.py, prompts.py) against production-use criteria.

## Summary

The tool is functionally sound but has **9 real bugs** (P0/P1) that can corrupt output, race under concurrency, or silently lose settings — plus 6 polish items.

## P0 — real bugs that will bite real users

### 1. `plan_indices` silently short-delivers when `exclude` is large
**File:** `prompts.py` in `plan_indices`
**Scenario:** after ~200+ rejected/used slots, the `+100` oversample headroom can't absorb exclude filtering, so `plan_indices` returns fewer than `count` items. `zip(slots_to_fill, plain_jobs)` in `run_job` silently drops unmatched slots — user sees "batch complete" but has gaps on disk.
**Fix:** aggressive oversample plus iterative top-up until `count` items collected or capacity exhausted.

### 2. `_poll_validation` raises TclError on destroyed window
**File:** `forge.py` ReviewWindow
**Scenario:** user closes review window while validation thread is still running. Thread keeps pushing to the queue; the poll's `DONE`/`ERR` branches call `.set_enabled()` on a destroyed widget → uncaught `TclError`.
**Fix:** wrap widget operations in `try/except tk.TclError`.

### 3. `scan_characters` crashes on `PermissionError`
**File:** `forge.py:scan_characters`
**Scenario:** any inaccessible subfolder under dataset root (OneDrive sync-locked, system folder, ACL restriction) raises `PermissionError` from `iterdir()`. Scan crashes mid-populate, GUI state inconsistent.
**Fix:** wrap each subfolder iteration in try/except, skip with log.

### 4. `apply_back` data race on `char.distribution`
**File:** `forge.py:CharRow.apply_back`
**Scenario:** `apply_back` is called from `_open_review` / `_open_prompt_preview` / `_edit_distribution` on the main thread. These can fire during generation while `run_job` worker thread reads `char.distribution`. Both mutate/read a shared Python object with no lock.
**Fix:** snapshot chars with `deepcopy` at `_start_job` time so workers see immutable state.

### 5. `DistributionEditor._clear` doesn't trigger auto-save
**File:** `forge.py:DistributionEditor._clear`
**Scenario:** user clicks "clear (use random)" → `char.distribution = None` is set but no Tk var is touched → no trace fires → no save. Close app → old distribution persists in `_settings.json` → next launch restores stale distribution.
**Fix:** explicitly fire the change callback.

### 6. `_edit_distribution` calls `apply_back` before opening editor
**File:** `forge.py:ForgeApp._edit_distribution`
**Scenario:** the pre-editor `apply_back` wipes distribution if count_var drifted from distribution sum (e.g., user used "apply to all" button). User opens editor to inspect current dist and finds it blank.
**Fix:** don't call apply_back before editor opens; editor reads directly from char.distribution.

### 7. `bind_all("<MouseWheel>")` conflicts across windows
**File:** `forge.py:PromptPreviewWindow._build_ui` and `ForgeApp._build_ui`
**Scenario:** PromptPreviewWindow's `canvas.bind_all` hijacks the global mousewheel binding. When the window closes, the binding is never restored — main list scroll becomes permanently broken for the session.
**Fix:** use `widget.bind` scoped to the canvas, not `bind_all`.

### 8. Review button has no interlock with active generation
**File:** `forge.py:ForgeApp._open_review`
**Scenario:** user clicks review while generation is running. Review window reads manifest concurrently with worker writes (atomic writes prevent corruption, but reviews are stale). If user clicks "apply & close" to delete a slot N's files, a worker may be actively writing N's PNG/TXT → file race.
**Fix:** disable review (and distribution, prompt preview, export) buttons during generation.

### 9. `engine._face_bytes` is a concurrency smell
**File:** `forge.py:run_job` sets `engine._face_bytes`/`engine._body_bytes` as mutable instance state
**Scenario:** not a live race today (char loop is sequential, pool context manager joins before next char). But the pattern is fragile — any future change that overlaps char iterations breaks invisibly.
**Fix:** pass bytes explicitly to `_run_one` via args instead of instance state.

## P1 — robustness

### 10. Distribution change mid-dataset causes silent category mismatch
**File:** `forge.py:run_job` with `char.distribution`
**Scenario:** user generates with dist=A, changes to dist=B, rejects some slots, hits regen. Refilled slots use B's category per `slot_to_category`, while kept slots have A's category recorded. Dataset ends up with mixed/inconsistent categorization per slot number. No visible warning.
**Fix:** when refilling, prefer manifest-recorded category over current distribution derivation.

### 11. Orphan `.json.tmp` files on disk-full
**File:** `forge.py:save_manifest` / `save_settings`
**Scenario:** disk full during `tmp.write_text` → tmp left on disk. Repeat → accumulating cruft.
**Fix:** `try/except` with `unlink(missing_ok=True)` on failure.

### 12. `apply_default_count` silently wipes per-character distributions
**File:** `forge.py:ForgeApp._apply_default_count`
**Scenario:** user carefully configures per-char dist like `15·5·5·5`. Clicks "apply to all" with default=30. All count_vars set to 30. No apply_back runs immediately, but next generate fires `apply_back` → count 30 mismatches dist sum 30 (wait, sums to 30, OK). Actually `30 == 15+5+5+5 == 30`. No wipe. Good.
But if user sets count to 40 via apply_to_all when dist=30: mismatch → dist wiped silently on Generate.
**Fix:** warn before applying or preserve distribution by scaling proportionally.

### 13. `FilePicker._set_path` leaked as semi-private
**File:** `forge.py:FilePicker._set_path` called from `ForgeApp._pick_face_for/_pick_body_for`
**Scenario:** convention violation, brittle to refactor.
**Fix:** rename to `set_path` (public).

### 14. Settings save failures are silent
**File:** `forge.py:ForgeApp._do_save`
**Scenario:** save fails (disk full, permission) → exception swallowed, user doesn't know settings weren't saved.
**Fix:** surface to log once per failure type; don't spam.

### 15. Log widget grows unbounded
**File:** `forge.py:ForgeApp._log`
**Scenario:** a 500-image run produces ~500 log lines. Not critical but polish.
**Fix:** truncate to last N lines when over cap.

## P2 — polish

### 16. API key plaintext in `_settings.json` on Desktop
Already mitigated by opt-in toggle. Reviewer notes desktop location could be OneDrive-synced. Acceptable given opt-in.

### 17. `run.bat` doesn't check plain `venv/` path
**Fix:** also probe `venv/Scripts/python.exe`.

### 18. Progress bar / status not reset after stop
Cosmetic — stale display until next Generate.

### 19. Pricing may drift
Already documented in README and tooltip.
