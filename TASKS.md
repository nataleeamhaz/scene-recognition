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

## Done: Claude Vision re-integration for recommendations
Reopened the cost/capability tradeoff from the CLIP/Haiku migration above,
scoped narrowly: CLIP (style/mood) and k-means (color palette) stay as the
free, deterministic priors, but `recommendations.py`'s Haiku call now also
receives the actual (resized) room photo, since neither CLIP nor YOLO can
see decor/textile items outside their fixed label sets (blanket, rug, wall
art, ...) or describe *where* something is. Decision made 2026-09-13:
replace the text-only call rather than run a second call alongside it, and
use `claude-haiku-4-5` (cheapest vision-capable tier) rather than Sonnet.

- [x] `recommendations.py::generate_recommendations()` now takes an
  `image_bytes` param, base64-encodes it, and sends it as an `image` content
  block alongside the existing facts-based text prompt.
- [x] `build_prompt()` (still pure/testable, no model call) updated to tell
  the model the attached photo is the real room, to notice what the fixed
  CLIP/YOLO categories miss, and to ground each recommendation in a specific
  visible item/color/position where the room supports it.
- [x] `main.py` threads the resized image through via `cv2_to_bytes()`
  (previously defined in `preprocessing.py` but unused — leftover from the
  pre-CLIP-migration Claude Vision era, and conveniently already tuned:
  `validate_and_resize()`'s 1568px cap is exactly Claude Vision's recommended
  max dimension).
- [x] Full suite still 32/32 (only `build_prompt`'s text output changed,
  covered by existing substring assertions).
- [x] Live smoke-tested against a real photo — recommendations now name
  specific visible items by color/position (e.g. "replace the dark
  bedspread with a...", "the dark geometric patterned rug") instead of
  generic prose.
- [ ] Not yet committed to git.

## Done: Final-photo generation (Decor8 AI) + site redesign
Two asks from the 2026-09-14 session: (1) generate a photorealistic "after"
photo showing the recommendations applied, and (2) redesign `static/index.html`
to feel calm, minimal, and trustworthy rather than generic.

**Final-photo generation:**
- Vendor decision: Decor8 AI, chosen over MeltFlex AI after checking both
  vendors' actual API docs (not just marketing pages) — Decor8 has
  transparent pay-as-you-go pricing ($0.20/image, no subscription) but its
  `generate_designs_for_room` endpoint only accepts a **publicly-reachable
  image URL**, not base64/file upload (confirmed directly against
  api-docs.decor8.ai). MeltFlex accepts base64 directly but requires an
  active paid subscription just to get an API key, with no published
  per-image USD price and no free trial.
- [x] New `photo_generation.py` — `resolve_design_style()` maps CLIP's
  `STYLE_LABELS` to Decor8's `design_style` enum (all 12 CLIP labels have a
  clean match, confirmed against Decor8's full enum list), `ROOM_TYPES` is a
  curated subset of Decor8's 31 room types, `generate_redesigned_photos()`
  calls the API and degrades gracefully to `[]` (same pattern as
  `pinterest.py`).
- [x] `main.py`: since Decor8 needs a public URL, added a short-lived
  in-memory temp-image store + `GET /tmp-image/{id}.jpg` for Decor8's
  servers to fetch the uploaded photo from. New `POST /generate-photo`
  endpoint, kept separate from `/analyze` (own rate limiter,
  `PHOTO_RATE_LIMIT_MAX_REQUESTS`, default 5/hour) since this is a slower
  (10-30s) real-money call the user opts into explicitly after already
  seeing text recommendations. New `GET /room-types` for the picker.
  Requires a new `PUBLIC_BASE_URL` env var — this app's actual
  internet-reachable address.
- [x] `static/index.html`: new "See It Applied" section — disclosure text
  (this is an AI-rendered approximation, not a guaranteed outcome), a
  room-type picker, "Generate final photo" button, its own loading/error
  states.
- [x] Tests: `tests/test_photo_generation.py` — style/room-type mapping,
  response parsing, guard clauses. Full suite: 39/39 passing.
- [x] Smoke-tested via `TestClient` with the real Decor8 call mocked out —
  confirmed the 503 fires correctly when `PUBLIC_BASE_URL` is unset, and
  that once set, `/tmp-image/{id}.jpg` correctly serves back the exact
  bytes at the URL Decor8 would be given.
- [ ] **Not yet live-tested against the real Decor8 API** — needs
  `DECOR8_API_KEY` in `.env`, plus `PUBLIC_BASE_URL` set to something
  actually internet-reachable (a deployed URL, or a local tunnel like
  ngrok) since plain `localhost` cannot be fetched by Decor8's servers.
- [ ] Room-type currently has no smart default (always defaults to the
  first option, `livingroom`) — there's no room-type classifier in the
  pipeline to infer it from. Revisit if this proves annoying in practice.

**Site redesign:**
- [x] Full visual rework of `static/index.html` — warm neutral palette
  (cream background, deep forest green + muted clay accents replacing the
  old indigo/purple), Fraunces serif headline paired with Inter for body
  text (via Google Fonts), softer border-based cards instead of heavy
  shadows, numbered rank badges for recommendations (CSS counters, no JS
  change), a minimal inline-SVG upload icon replacing the house emoji.
- [x] All existing element IDs/classes the JS relies on kept unchanged —
  only CSS and non-JS-driven markup changed; `<script>` logic untouched
  except for wiring the new generate-photo button.
- [x] Verified live in a real browser (Playwright + Chromium, installed
  fresh into `.venv` this session — `chromium-cli` wasn't available and
  GUI automation on this machine didn't produce visible windows for
  `screencapture`). Full `/analyze` flow run end-to-end against the real
  APIs: upload → preview → quality warning → furniture chips → style card
  → recommendations → generate-photo's 503 error state. Zero console
  errors at every step. Screenshots confirmed the design renders as
  intended.
- [ ] Not yet committed to git.

## Backlog (rough priority order)
1. **Push the CLIP/Haiku migration commit** — committed locally (`07c922d`)
   but not yet pushed to GitHub.
2. [x] **Hide furniture confidence scores from the UI** — removed the `.conf`
   span from `static/index.html`'s chip rendering (and the now-unused `.conf`
   CSS rule); chips show label only. `feature_extraction.py` still returns
   confidence internally, unchanged.
3. **Recommendations: rank by importance** — [x] done (2026-08-28).
   Decided against per-recommendation images for now (see note below);
   scope narrowed to ranking + layout reorder.
   - [x] `recommendations.py::build_prompt()` now asks Haiku for an array of
     5 `{rank, text}` objects instead of 5 plain strings, ordered by rank
     ascending (1 = most important).
   - [x] `static/index.html` sorts `sa.recommendations` by `rank` client-side
     before rendering (defensive against the model returning them out of
     order) and renders `.text`.
   - [x] Page reorder: pulled the recommendations list out of the Style
     Analysis card into its own "Recommendations" section, placed *after*
     Pinterest Inspiration — images now render above the text
     recommendations, per direction from the 2026-08-28 session.
   - [ ] **Deferred, not rejected: per-recommendation example images.**
     Considered extending `search_multiple_terms()` to run one Pinterest
     search per recommendation (~5 calls/request instead of 1) so each rec
     gets a matching image. Retracted for now in favor of keeping the single
     shared Pinterest Inspiration grid — revisit if the shared grid doesn't
     give recommendations enough visual grounding in practice.
4. **Feedback loop / iteration** — show a final example and let the user give
   feedback to refine it further, rather than one-shot analyze-and-done.
   Needs design: does feedback re-run the full pipeline, or just a follow-up
   Haiku call reusing the same style/mood/color/furniture context? Multi-turn
   state (conversation history, or at minimum the prior analysis result) needs
   to persist across the feedback round-trip — currently `/analyze` is fully
   stateless per request.
5. **Small-space-aware recommendations** — nothing currently encodes room
   size/constraints; PRD specifically calls out small NY apartments.
6. **Spatial furniture arrangement** — partially addressed by the Claude
   Vision re-integration above (recommendations can now reference *where*
   an item is in prose, e.g. "move the armchair from the corner to face the
   window"). Still not an actual layout/arrangement diagram or coordinate
   output — that remains a stretch goal.
7. **Pinterest curation** — trial-mode API only searches your own account's
   saved pins (see `pinterest.py` docstring). Need to seed a board with ~20-30
   pins for results to actually be "curated" rather than empty.
8. **`feature_extraction.py` has no tests** — the one module left without
   coverage (matches the existing pattern of not unit-testing model-loading
   integration code, but worth a second look — e.g. testing the dedup-by-
   confidence logic with a mocked YOLO result).
9. [x] **Dead code cleanup** — removed `preprocessing.py`'s standalone
   `FastAPI()` app and `/upload` route; the module now only exports the
   functions `main.py` actually imports. Full test suite (27/27) still
   passes.
10. [x] **Board-as-inspiration-source, single-tenant** — done (2026-08-28),
    scoped down from the original "user-linked" framing after a design
    discussion. Decision: trial-mode Pinterest API can only ever reach the
    *authenticated account's own* boards/pins anyway, and the app has no
    session/multi-user infrastructure — so "linking a board" for now means
    picking one of the configured account's own boards, not per-visitor
    OAuth. True multi-user board linking (each end user authorizing their
    own separate Pinterest account) is a much bigger follow-up: needs
    session/token storage and likely Pinterest standard API approval.
    - [x] `pinterest.py`: added `list_boards()` (`GET /v5/boards`) and
      `get_board_pins(board_id)` (`GET /v5/boards/{id}/pins`), sharing a new
      `_parse_pin_item()` helper with `search_pins()`.
    - [x] `main.py`: new `GET /pinterest/boards` endpoint; `/analyze` takes
      an optional `board_id` form field — when set, `pinterest_results`
      comes from that board's pins instead of Haiku's keyword search
      (board choice wins, since it's an explicit "this is what I'm
      envisioning" signal from the user).
    - [x] `static/index.html`: board picker `<select>` (hidden if the
      account has no boards), populated from `/pinterest/boards` on load;
      selected `board_id` sent along with `/analyze`.
    - [x] Tests added in `tests/test_pinterest.py` for the new functions'
      guard clauses and `_parse_pin_item` image-fallback logic. Full suite:
      32/32 passing.
    - [x] Verified end-to-end against the real API — confirmed the code
      degrades gracefully (empty results, no crash) rather than erroring.
    - [ ] **Found during testing, not fixed here**: `PINTEREST_ACCESS_TOKEN`
      in `.env` is currently expired/invalid (raw API returns 401). Needs a
      fresh token via `/pinterest/login` → `/callback` before board
      selection (or the existing keyword search) can return real results.
    - [ ] Deferred: using board pins to *influence* recommendations (item 11
      below) — this pass is display/selection only, per the 2026-08-28
      design discussion.
11. **Use loaded Pinterest boards to inform suggestions** — beyond just
    displaying board pins back to the user, use them as reference/training
    signal (e.g. embed board pin images and factor their style/color into
    what `recommendations.py` generates) so suggestions lean toward the
    user's own visual taste, not just the generic CLIP/Haiku pipeline output.
    Needs design: embedding approach, and how this composes with the
    deferred CLIP-based visual Pinterest matching (item 3, "Deferred" note).

## Testing approach
- Unit-test each new function in isolation (no network/model calls) before
  wiring it into the pipeline.
- Manual E2E in the browser once a feature is wired into `/analyze`.
- Test files land alongside the module they cover as it's built, not as one
  big pass at the end.
