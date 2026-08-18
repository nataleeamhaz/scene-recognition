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

## Done: CLIP/Haiku hybrid style analysis
Replaced the single Claude Sonnet Vision call with a hybrid pipeline —
resolves the open question from the previous entry (kept a small LLM call
for the two fields CLIP can't produce, rather than templating them).

- [x] `style_analysis.py` — CLIP (`openai/clip-vit-base-patch32` via
      `transformers`) zero-shot classification for `style`/`mood`, k-means
      (via `cv2.kmeans`, no new dependency) for `color_palette` as hex codes.
- [x] `recommendations.py` — `claude-haiku-4-5`, text-only (no image), for
      just `existing_strengths` + `recommendations` + `pinterest_search_terms`.
      ~4x cheaper per call than the old full-vision Sonnet call (see cost
      breakdown from the migration discussion — image removal + a smaller
      output schema + a cheaper model tier all contributed).
- [x] `main.py` rewired: preprocess → quality → YOLO → CLIP+k-means →
      Haiku → Pinterest. `ANTHROPIC_API_KEY` is still needed (Haiku), just
      no longer for vision.
- [x] `static/index.html`'s `colorToCSS` updated — color palette entries are
      now hex codes from k-means, not color-name strings.
- [x] Unit tests: `tests/test_style_analysis.py` (`pick_best_label`,
      `extract_color_palette`), `tests/test_recommendations.py`
      (`build_prompt`). Full suite: 27/27 passing. Full `main.py` import
      verified end-to-end (triggers CLIP checkpoint + YOLO weight download).
- [ ] **Manual browser E2E still needed** — run `uvicorn main:app --reload`,
      upload a real photo, confirm the full pipeline produces sane output.
- [ ] **Deployment weight**: `torch` + `transformers` + the CLIP checkpoint
      (~600MB) add real bulk (~2GB+ total with `ultralytics`/`torch`) to what
      was a lightweight app — reconsider the Render free-tier plan (disk/RAM
      limits, build time) before actually deploying.
- [ ] **Deferred**: CLIP-based visual Pinterest matching (embed pin images,
      rank by similarity to the room photo) — a real upgrade over today's
      keyword search, but scoped out of this pass since it needs to fetch
      and embed a dynamic image set per request.
- [ ] Not yet committed to git.

## Backlog (rough priority order)
1. **Commit the CLIP/Haiku migration** — `style_analysis.py`,
   `recommendations.py`, the `main.py` rewire, `requirements.txt`, and the
   `static/index.html` hex-color fix are all local-only right now.
2. **Small-space-aware recommendations** — nothing currently encodes room
   size/constraints; PRD specifically calls out small NY apartments.
3. **Spatial furniture arrangement** — current output is prose recommendations,
   not an actual layout/arrangement suggestion. Stretch goal.
4. **Pinterest curation** — trial-mode API only searches your own account's
   saved pins (see `pinterest.py` docstring). Need to seed a board with ~20-30
   pins for results to actually be "curated" rather than empty.
5. **`feature_extraction.py` has no tests** — the one module left without
   coverage (matches the existing pattern of not unit-testing model-loading
   integration code, but worth a second look — e.g. testing the dedup-by-
   confidence logic with a mocked YOLO result).
6. **Dead code cleanup** — `preprocessing.py` still defines its own `FastAPI()`
   app and `/upload` route, disconnected from `main.py`'s real app. Remove.

## Testing approach
- Unit-test each new function in isolation (no network/model calls) before
  wiring it into the pipeline.
- Manual E2E in the browser once a feature is wired into `/analyze`.
- Test files land alongside the module they cover as it's built, not as one
  big pass at the end.
