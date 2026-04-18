# Fix Plan — Second Pass

Consolidated from my audit + independent reviewer findings.

## P0

1. **Files API ref cleanup on error paths.** Extract `_cleanup_ref_files(engine, names)` and call on every early-exit in `run_batch_submit` (including the `not key_map`, JSONL upload fail, batch submit fail cases) and in non-success branches of `run_batch_poll`.
2. **Stream-download batch results.** Download to a temp file, then stream-iterate lines. Avoids holding 500MB+ of results in memory for large batches.
3. **Fix progress bar total for batch.** Update `pbar.configure(maximum=...)` to `len(key_map)` after JSONL build completes. Also reset `_total_jobs` and `_char_done` at that point.

## P1

4. **MIME type hygiene.** Change `mime_type="jsonl"` → `"application/x-ndjson"` for the batch JSONL upload.
5. **Stop-event race between submit and poll threads.** Emit `batch_submitted` message AFTER the submit thread is fully done (last line before implicit return). Don't `stop_event.clear()` inside `_spawn_batch_poll` — the submit thread already cleared it in `_start_job`.
6. **Resume-on-launch initializes progress state.** Pre-populate `_total_jobs`, `_done_jobs`, and `_char_done` from the loaded batch state so the UI shows live counts when polling resumes.
7. **Stop-event checks in `build_batch_jsonl`.** Check between character iterations so a Stop during long multi-char uploads actually aborts.
8. **Resume + no API key.** Detect empty API key in `_maybe_resume_batch`; prompt user to enter the key and retry resume, rather than silently giving up.
9. **Skip re-write if file already exists on resume.** In result-processing loop, if `NNN.png` + `NNN.txt` already exist (previous session partially processed), skip — preserves any validation scores.

## Execution order

1. Helper extraction: `_cleanup_ref_files`
2. `run_batch_submit`: wire cleanup into all exit paths + fix MIME + remove trailing done message semantics
3. `run_batch_poll`: stream download + skip-if-exists + cleanup on failure branches
4. `_spawn_batch_poll`: don't clear stop_event if submit thread may be running
5. `_maybe_resume_batch`: handle missing key + init progress
6. `build_batch_jsonl`: accept stop_event, check between chars
7. `_handle_msg` for `batch_submitted`: set pbar max and reset progress state
8. Compile + smoke test

All fixes verified with the existing smoke-test patterns.
