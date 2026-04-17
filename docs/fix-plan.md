# Fix Plan

Execute in this order. Each step is isolated (no cross-dependencies).

## P0 fixes

1. **prompts.py: `plan_indices` robust sampling**
   - Raise oversample to `count*20 + 500`, clamp to total capacity
   - Add iterative top-up loop that re-samples from advanced RNG state until count satisfied or total exhausted

2. **forge.py: `scan_characters` permission tolerance**
   - Wrap each `sub.iterdir()` in try/except PermissionError/OSError
   - Log skipped dirs, don't crash scan

3. **forge.py: `_poll_validation` widget safety**
   - Wrap widget ops (`set_enabled`, `configure`) in try/except TclError
   - Exit loop cleanly if widget gone

4. **forge.py: snapshot chars at generation start**
   - Use `copy.deepcopy([r.apply_back() for r in rows])` in `_start_job`
   - Workers receive immutable snapshot; main thread can edit rows freely

5. **forge.py: `DistributionEditor._clear` triggers save**
   - Call `self.row._fire_change()` after setting distribution=None

6. **forge.py: `_edit_distribution` stops calling apply_back**
   - Remove the pre-editor apply_back call (distribution editor reads char.distribution directly)

7. **forge.py: scoped mousewheel in PromptPreviewWindow**
   - Replace `canvas.bind_all("<MouseWheel>", ...)` with `canvas.bind(...)` + binding on inner frame
   - Same for any other Toplevel that scrolls

8. **forge.py: interlock — disable review/dist/preview/export during generation**
   - Track all these buttons in a list
   - Disable on `_start_job`, re-enable on `done` msg

9. **forge.py: pass bytes explicitly through `_run_one`**
   - Add `face_bytes`/`body_bytes` params to `_run_one`
   - Remove `engine._face_bytes`/`engine._body_bytes` writes in run_job
   - Load per-char bytes in run_job outer loop, pass to pool.submit

## P1 fixes

10. **forge.py: manifest-recorded category preferred on refill**
    - In run_job distribution branch, check existing manifest entry first for category
    - Only fall back to `slot_to_category(slot, char.distribution)` if no manifest record

11. **forge.py: save_manifest/save_settings tmp cleanup**
    - try/except around write, unlink tmp on failure, re-raise

12. **forge.py: FilePicker `set_path` public**
    - Add `set_path` public alias that wraps `_set_path`
    - Update `_pick_face_for`/`_pick_body_for` to use public API

13. **forge.py: surface save failures**
    - Log once with error tag; suppress repeats within a session via flag

14. **forge.py: log truncation**
    - After each insert, if lines > 2000, delete oldest 500
    - Keeps log responsive and bounded

15. **forge.py: apply_default_count preserves dist**
    - If char has distribution AND new count matches dist sum, leave alone
    - If count differs, warn user once per batch that dist will be dropped

## P2 fixes

16. **run.bat: also probe `venv/Scripts/python.exe`**

17. **forge.py: reset progress on stop**
    - On "done" msg after stop, reset pbar value to 0 and status to "idle"

## Verification

After each P0/P1 fix:
- `python -m py_compile forge.py prompts.py`
- Full GUI init smoke test (existing pattern)

Final:
- End-to-end smoke: scan → dist edit → clear dist → settings round-trip → start fake job (dry-run compatible API?) → verify manifest preserves category → review window open/close lifecycle
