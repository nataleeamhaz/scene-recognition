# Room Design Recommender

Upload a photo of a room and get furniture detection, style/mood analysis, and
AI-generated design recommendations, plus matching inspiration pulled from
Pinterest.

## Architecture

```mermaid
flowchart TD
    A[["📷 Room photo upload"]] -->|"POST /analyze"| B["preprocessing.py"]

    B --> Q["assess_image_quality()<br/>blur / resolution / exposure"]
    B --> R["validate_and_resize()"]

    R --> YOLO["YOLO — yolov8n.pt<br/><i>local, COCO-pretrained</i>"]
    R --> CLIP["CLIP — clip-vit-base-patch32<br/><i>local, zero-shot style/mood</i>"]
    R --> KM["k-means<br/><i>local, dominant colors</i>"]

    YOLO -->|"detected_furniture"| HAIKU
    CLIP -->|"style, mood"| HAIKU
    KM -->|"color_palette"| HAIKU
    R -->|"resized room photo"| HAIKU
    A -.->|"optional text prompt"| HAIKU

    HAIKU["Claude Haiku 4.5 (vision)<br/><i>recommendations,<br/>strengths, search terms</i>"]

    HAIKU -->|"pinterest_search_terms"| PIN["Pinterest API<br/><i>inspiration images</i>"]

    Q --> RESP
    YOLO --> RESP
    HAIKU --> RESP
    PIN --> RESP
    RESP["JSON response:<br/>quality + furniture + style_analysis + pinterest_results"]

    style YOLO fill:#e8f0fe,stroke:#5b7fd6
    style CLIP fill:#e8f0fe,stroke:#5b7fd6
    style KM fill:#e8f0fe,stroke:#5b7fd6
    style HAIKU fill:#fdeee0,stroke:#d68b3a
    style PIN fill:#fdeee0,stroke:#d68b3a
```

Blue steps run locally with no per-request API cost. Orange steps call an
external API. YOLO, CLIP, and k-means run independently off the same
preprocessed image — Haiku is the only step that combines all of their
outputs (plus the user's optional text prompt) into the final recommendations
and Pinterest search terms.

### Pipeline stages

1. **`preprocessing.py`** — decodes the upload, runs objective quality checks
   (blur, resolution, exposure), and resizes for the downstream models.
2. **YOLO** (`feature_extraction.py`) — detects furniture/objects and returns
   labeled, deduplicated bounding-box confidences.
3. **CLIP + k-means** (`style_analysis.py`) — CLIP zero-shot classifies style
   and mood against fixed candidate labels; k-means clusters pixels for the
   dominant color palette. Both are local, free, and run on every request.
4. **Claude Haiku, vision** (`recommendations.py`) — the only generative step;
   sees the resized room photo directly alongside the structured facts from
   steps 2–3 and the user's prompt, so recommendations can reference specific
   visible items, colors, and positions that CLIP/YOLO's fixed label sets miss.
5. **Pinterest** (`pinterest.py`) — searches using Haiku's generated terms for
   inspiration images.

## Running locally

```
source .venv/bin/activate
uvicorn main:app --reload
```

Then open `http://localhost:8000`.
