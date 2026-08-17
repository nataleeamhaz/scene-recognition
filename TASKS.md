# Remaining Tasks

Tracking gaps against the PRD ("Bedroom Design PRD" in Drive). Working one item
at a time, testing each component before layering the next feature on top.

## Done: image quality feedback (first pass)
PRD ask: "is it too blurry, what angles should I get."

- [x] Objective CV checks in `preprocessing.py::assess_image_quality()` — blur
      via Laplacian variance, minimum resolution, over/under exposure via mean
      brightness. Pure function, no API calls.
- [x] Decision: warn-and-continue, not block-and-reject. `/analyze` always
      runs the full pipeline; `quality.warnings` (list of strings) rides along
      in the response. Never blocks on a borderline threshold.
- [x] Wired into `main.py`'s `/analyze` — quality check runs on the *original*
      decoded image, before `validate_and_resize()` (downscaling smooths out
      the high-frequency detail the blur check needs).
- [x] Surfaced in `static/index.html` — amber warning banner above the results
      section, rendered from `data.quality.warnings` when non-empty.
- [x] Unit tests in `tests/test_preprocessing.py` (10 tests, all passing) —
      covered before wiring into `/analyze`, per the incremental-testing plan.
- [ ] Thresholds (`blur_threshold=100`, `min_dimension=400`,
      `dark_threshold=40`, `bright_threshold=220`) are untuned guesses — revisit
      against real room photos once there's a way to test with real uploads.
- [ ] Angle/framing feedback (deferred) — needs semantic judgment, not just
      pixel stats. Revisit once the CLIP migration (below) settles how/whether
      any generative text still happens in this pipeline.
- [ ] Manual E2E in browser still needed (start `uvicorn main:app --reload`,
      upload a real blurry/dark/low-res photo, confirm the banner renders).

## Backlog (rough priority order)
2. **Migrate style analysis from Claude Vision to CLIP** — `main.py`'s
   `claude_analyze()` currently calls Anthropic's Claude Vision to produce the
   whole `style_analysis` JSON. Decision made: replace it with CLIP entirely.
   - `style` / `mood` → zero-shot classification (cosine similarity between
     image embedding and candidate label-text embeddings)
   - `color_palette` → plain CV (k-means on pixels), not CLIP
   - **Open question**: `recommendations` (5 actionable improvements) and
     `existing_strengths` are generative text — CLIP can't produce these.
     Need a decision: template/rule-based text from classification results,
     or keep a small LLM call just for phrasing.
   - Once migrated, `ANTHROPIC_API_KEY` / the `anthropic` dependency likely
     go away entirely.
3. **Small-space-aware recommendations** — nothing currently encodes room
   size/constraints; PRD specifically calls out small NY apartments.
4. **Spatial furniture arrangement** — current output is prose recommendations,
   not an actual layout/arrangement suggestion. Stretch goal.
5. **Pinterest curation** — trial-mode API only searches your own account's
   saved pins (see `pinterest.py` docstring). Need to seed a board with ~20-30
   pins for results to actually be "curated" rather than empty.
6. **Test suite** — no tests exist yet. Add pytest coverage per module as it's
   touched: `preprocessing.py`, `feature_extraction.py`, `pinterest.py`, and
   whatever replaces `claude_analyze()` after the CLIP migration.
7. **Dead code cleanup** — `preprocessing.py` still defines its own `FastAPI()`
   app and `/upload` route, disconnected from `main.py`'s real app. Remove.
8. **Commit current work** — `main.py`, `feature_extraction.py`, `pinterest.py`,
   `static/`, `requirements.txt` are untracked; only `preprocessing.py` has a
   pending diff.

## Testing approach
- Unit-test each new function in isolation (no network/model calls) before
  wiring it into the pipeline.
- Manual E2E in the browser once a feature is wired into `/analyze`.
- Test files land alongside the module they cover as it's built, not as one
  big pass at the end.
