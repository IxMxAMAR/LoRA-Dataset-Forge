# Fix Plan — Third Pass

## Executed

- **P0** Path traversal via `char.name` — `_sanitize_char_name` helper + resolved-path escape check in `scan_characters`
- **P0** README pipeline diagram `48 framings` → `10 framings`
- **P0** Training-config export correctness — verified against upstream docs:
  - ai-toolkit promoted to primary target (LO confirmed actual usage)
  - `resolution: [ 512, 768, 1024 ]` (multi-res bucket, verified against `config/examples/train_lora_qwen_image_24gb.yaml`)
  - musubi-tuner command updated to include required `--dit`, `--vae`, `--text_encoder` placeholders + correct script path `src/musubi_tuner/qwen_image_train_network.py`
  - Added RTX 5090 / 32GB VRAM hyperparameter guidance (rank 32, batch 2, resolution [768,1024,1344])
- **P1** `_scan_and_populate` now gated by `_job_running` (matches all other destructive actions)
- **P1** Case-insensitive name collision filter (`seen_casefolded` set in scan_characters)
- **P1** Caption double-trigger — removed the second `of {trigger}` occurrence; LoRA conventions use one trigger at start
- **P1** Outfit/scene coherence hint added to `SYSTEM_PROMPT` — handles degenerate combinations like "trench coat on yoga mat" gracefully
- **P1** `_preview` window deduplication — track per-char Toplevel, `lift()` existing instead of leaking
- **P2** `dataclasses.field` unused import removed
- **P2** f-string-without-placeholders cosmetic fixes (~7 spots)
- **P2** Unused `total` local in `_aspect_table`

## Deferred (acceptable risk / out of scope)

- Toggle trace cleanup on rescan — CPython refcounting clears it deterministically; real-world risk low
- Streaming download for batch results — SDK doesn't expose streaming; documented memory limit
- ReviewWindow synchronous 500-thumbnail render — add lazy loading if users report long open times
- Near-duplicate lightings in pool — cosmetic diversity reduction, doesn't cause failure
- `bind_all` mousewheel conflict when two Toplevels open — realistic usage opens one at a time

## Verification

- `pyflakes forge.py prompts.py` — clean
- `py_compile` — clean
- Path sanitization smoke tests: `../exfil` → `exfil`, `..evil` → `evil`, `a/b/c` → `abc`
- Caption: trigger word appears exactly once (was 2 before)
- Exported ai-toolkit YAML uses multi-res `[ 512, 768, 1024 ]` format
