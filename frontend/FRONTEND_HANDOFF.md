# Frontend Instruction Set For MRZ Upload And Alignment App

## Summary

Build a frontend app in React or Angular that lets a user:

1. Upload a passport photo or PDF.
2. Fetch and display the generated preview image from the backend.
3. Let the user correct document orientation and fine alignment in the browser.
4. Send the final alignment values to the API for extraction.
5. Show the returned MRZ lines, parsed fields, status, duration, and report link.

The frontend should not try to parse MRZ text, normalize names, or "fix" extracted content. Its job is to help the user produce a better aligned image for backend extraction.

Use `opencv.js` as the primary frontend image-transform engine. Use plain canvas rendering only as a fallback for preview/UI continuity when `opencv.js` is unavailable. This matters because pure canvas coordinates and rendering behavior do not always translate cleanly into the backend OpenCV crop/transform pipeline.

## Project Scope

- Target only TD3 passport MRZ.
- Expect exactly 2 MRZ lines.
- Expect 44 characters per line.
- Do not design this first version around visas, ID cards, or other document types.

## Backend Contract

Observed API routes in this repo:

- `GET /api/health`
- `POST /api/uploads`
- `GET /api/documents/{document_id}/preview`
- `POST /api/extractions`
- `GET /api/extractions/{extraction_id}/report`
- `GET /api/references`
- `POST /api/references`

For the upload-and-extract flow, the frontend mainly needs these three:

### 1. Upload

Endpoint:

- `POST /api/uploads`

Request:

- `multipart/form-data`
- field name: `file`

Response shape:

```json
{
  "document_id": "string",
  "filename": "passport.png",
  "source_type": "image",
  "extension": ".png",
  "file_hash": "sha256-or-null",
  "deduplicated": false,
  "preview_width": 1200,
  "preview_height": 800
}
```

Frontend behavior:

- Save `document_id` in state.
- Use it as the primary key for all later actions.
- Reset alignment state after a new upload.

### 2. Preview

Endpoint:

- `GET /api/documents/{document_id}/preview`

Frontend behavior:

- Load this PNG after upload succeeds.
- Prefer rendering and transforming the preview through `opencv.js`.
- Use plain canvas only as a fallback display path.
- Use the preview image as the exact source for browser-side transform controls.

### 3. Extraction

Endpoint:

- `POST /api/extractions`

Recommended request body:

```json
{
  "document_id": "string",
  "rotation": 0,
  "transform": {
    "micro_rotation": 0.0,
    "zoom": 1.0,
    "offset_x": 0.0,
    "offset_y": 0.0,
    "viewport_width": 1200,
    "viewport_height": 800
  },
  "crop": {
    "x": 0.04,
    "y": 0.04,
    "width": 0.92,
    "height": 0.92
  },
  "use_face_hint": false
}
```

Response shape:

```json
{
  "extraction_id": "string",
  "status": "ok",
  "filename": "passport.png",
  "line1": "P<...",
  "line2": "...",
  "parsed": {},
  "duration_ms": 1234.56,
  "report_path": "storage/reports/....json",
  "document_id": "string"
}
```

## What The Frontend Must Own

The frontend should own only visual alignment controls and request assembly.

It should allow the user to:

- rotate in 90 degree steps: `0`, `90`, `180`, `270`
- apply fine micro-rotation
- zoom in and out
- pan horizontally and vertically
- keep the passport page centered inside a fixed working frame
- submit the transform state to the backend

Implementation preference:

- Primary: `opencv.js` for rotation, affine transform preview, crop overlay math, and coordinate mapping
- Fallback: HTML canvas for display-only interaction when `opencv.js` is not available

The frontend should not:

- perform OCR
- parse TD3 fields
- auto-correct MRZ text
- normalize line 1 names or separators
- silently rewrite extracted text

## Transform Rules

These rules come from the current repo behavior and should be preserved.

Important frontend policy:

- The browser transform math should mirror backend OpenCV behavior as closely as possible.
- Prefer `opencv.js` matrices and image operations over canvas-only transforms when generating or validating transform coordinates.
- Do not trust canvas display coordinates alone as the source of truth for crop mapping.

### Coarse Rotation

- Use `rotation` for 90-degree correction only.
- Allowed values: `0`, `90`, `180`, `270`.
- This is separate from fine alignment.

### Fine Transform

Send `transform` with:

- `micro_rotation`
- `zoom`
- `offset_x`
- `offset_y`
- `viewport_width`
- `viewport_height`

Notes:

- Positive browser rotation is visual clockwise.
- The backend internally converts this for OpenCV.
- Offsets are normalized ratios, not raw pixels.
- `viewport_width` and `viewport_height` should match the rendered viewer canvas.
- If the frontend computes transform previews with `opencv.js`, keep the same transform conventions when building the API payload.

### Crop

- Keep a fixed working crop in normalized coordinates.
- Current prototype uses:
  - `x = 0.04`
  - `y = 0.04`
  - `width = 0.92`
  - `height = 0.92`
- The backend applies this crop after rotation and transform rendering.
- Crop coordinates should be derived from the same transformed image space shown to the user, preferably via `opencv.js`.

Important:

- Do not make crop editing freeform in the first version unless requested.
- The current backend flow assumes a stable viewport crop and frontend-driven alignment.
- This fixed crop is safer than canvas-only user-drawn crop boxes because canvas coordinate translation has already shown mismatch risk with backend OpenCV cropping.

## Suggested UI Flow

### Screen 1: Upload

Required elements:

- file picker
- upload button
- loading state
- error state

After success:

- store `document_id`
- fetch preview
- move user into alignment screen

### Screen 2: Alignment

Required elements:

- preview canvas
- rotate left button
- rotate right button
- reset adjustments button
- micro-rotation slider
- zoom slider or zoom buttons
- horizontal offset slider
- vertical offset slider
- optional drag-to-pan on canvas
- extract button

Viewer rules:

- Prefer an `opencv.js`-driven viewer pipeline for transform preview.
- show a visible centered working frame overlay
- dim the area outside the working frame
- highlight the lower area where MRZ usually lives
- keep controls responsive while preview updates
- If `opencv.js` fails to load, degrade gracefully to canvas preview and clearly keep transform behavior conservative

### Screen 3: Result

Required elements:

- extraction status
- MRZ line 1
- MRZ line 2
- parsed identity fields when present
- parsed document fields when present
- extraction duration
- link to JSON report when available

Warnings to show:

- line 1 length is not 44
- line 2 length is not 44
- parsed document number missing
- parsed nationality missing
- parsed name fields missing

## State Model

Recommended frontend state:

```ts
type FrontendState = {
  upload: UploadResponse | null;
  documentId: string | null;
  previewUrl: string | null;
  rotation: 0 | 90 | 180 | 270;
  microRotation: number;
  zoom: number;
  offsetX: number;
  offsetY: number;
  viewportWidth: number;
  viewportHeight: number;
  isBusy: boolean;
  extractionResult: ExtractionResponse | null;
};
```

Recommended defaults:

- `rotation = 0`
- `microRotation = 0`
- `zoom = 1`
- `offsetX = 0`
- `offsetY = 0`
- `use_face_hint = false`

## React Or Angular Guidance

Either framework is fine. Pick one and keep the first version simple.

Whichever framework is chosen, the image-processing layer should be isolated behind a small adapter so the app can:

- use `opencv.js` as primary
- fall back to canvas rendering when needed
- keep one consistent payload builder for the backend

### If React

Suggested structure:

- `UploadPage`
- `AlignmentViewer`
- `TransformControls`
- `ExtractionResultPanel`
- `imageTransformAdapter`
- `apiClient`

Prefer:

- controlled state for transform values
- canvas rendering in a dedicated component
- `opencv.js` loading and transform logic behind a dedicated adapter/service
- one service module for API calls

### If Angular

Suggested structure:

- `upload` component
- `alignment-viewer` component
- `transform-controls` component
- `result-panel` component
- `image-transform` service
- `api` service

Prefer:

- typed interfaces for API contracts
- `opencv.js` wrapped in a dedicated service
- one shared state service if multiple components need the same document session

## UX Rules

- Show upload progress or at least a busy state.
- Disable extraction while upload or extraction is running.
- Reset transform state on new upload.
- Keep status messages plain and actionable.
- Preserve the last extraction result until a new extraction finishes or a new upload starts.
- Do not hide raw MRZ output behind only parsed fields.

## Error Handling

Handle these cases clearly:

- empty file upload
- unsupported file or decode failure
- unknown `document_id`
- preview not found
- extraction failure
- report not found

Show backend error `detail` when available.

## Data Integrity Rules

These are important for this repo:

- The backend is truth-first.
- The frontend must not beautify or normalize MRZ text.
- Display returned MRZ exactly as received.
- If parsed fields and raw MRZ appear inconsistent, keep both visible.
- Treat line 2 as structurally stronger, but do not override line 1 in the UI.

## What To Reuse From Existing Prototype

There is already a vanilla browser prototype in this repo:

- [frontend/app.js](/home/inam/scicom_dev/SCI-OCR/frontend/app.js)
- [frontend/index.html](/home/inam/scicom_dev/SCI-OCR/frontend/index.html)
- [frontend/styles.css](/home/inam/scicom_dev/SCI-OCR/frontend/styles.css)

Use it as behavior reference for:

- payload shape
- viewer interactions
- transform defaults
- result presentation

Do not copy it blindly. Port the behavior cleanly into React or Angular components.

Important:

- The current prototype is useful as a flow reference, but it is still canvas-centric.
- The new frontend should upgrade this by making `opencv.js` the primary transform engine so frontend alignment better matches backend OpenCV behavior.

## Delivery Checklist For Frontend Developer

- Implement upload using `POST /api/uploads`
- Load preview from `GET /api/documents/{document_id}/preview`
- Implement an `opencv.js`-based transform viewer
- Add canvas-only fallback when `opencv.js` is unavailable
- Support 90-degree rotation and fine transform controls
- Send extraction request to `POST /api/extractions`
- Show raw MRZ lines and parsed fields
- Show extraction errors and loading states
- Keep transform payload aligned with backend schema
- Do not add client-side MRZ correction logic
- Avoid deriving backend crop intent from canvas-only coordinates when `opencv.js` is available

## Observed Facts vs Inference

Observed from code:

- The backend expects `rotation`, `crop`, and `transform` in extraction requests.
- The frontend prototype already uses a fixed crop and canvas-based transform controls.
- The backend applies rotation, then transform, then crop before MRZ extraction.
- The backend returns both raw MRZ lines and parsed fields.

Inference:

- A React or Angular rewrite should preserve the existing request contract, but move the frontend transform engine closer to backend OpenCV semantics.

Recommended approach:

- Build the new frontend as a componentized version of the current prototype with better maintainability, keep the API contract unchanged, and use `opencv.js` as the primary transform layer.
