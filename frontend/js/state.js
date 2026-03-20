export const state = {
  upload: null,
  documentId: null,
  previewImage: null,
  scale: 1,
  rotation: 0,
  fineRotation: 0,
  offsetX: 0,
  offsetY: 0,
  dragPointer: null,
  dragOrigin: null,
  dragMode: null,
  viewBaseScale: 1,
  viewSize: { width: 720, height: 480 },
  isBusy: false,
  extractionResult: null,
  animationFrameId: null,
  animationStartMs: 0,
  guidance: {
    cvStatus: "loading",
    mrzRect: null,
    mrzScore: 0,
    mrzMisses: 0,
    faceRect: null,
    faceConfidence: 0,
    lastAnalysisMs: 0,
    lastFaceMs: 0,
    facePending: false,
  },
};

export const VIEW_MIN_WIDTH = 320;
export const VIEW_MIN_HEIGHT = 240;
export const CAPTURE_ASPECT_RATIO = 4 / 3;
export const ZOOM_MIN = 0.85;
export const ZOOM_MAX = 2.4;
export const MRZ_GUIDE_HEIGHT_RATIO = 0.18;
export const MRZ_GUIDE_WIDTH_RATIO = 0.92;
export const MRZ_GUIDE_BOTTOM_MARGIN_RATIO = 0.04;
export const ROTATE_DRAG_SENSITIVITY = 0.024;
export const QUALITY_SAMPLE_MAX_WIDTH = 180;
export const QUALITY_LOW_CONTRAST_STDDEV = 28;
export const QUALITY_LOW_EDGE_STRENGTH = 15;
export const GUIDANCE_ANALYSIS_INTERVAL_MS = 220;
export const FACE_ANALYSIS_INTERVAL_MS = 700;
export const GUIDANCE_LOCK_MISS_LIMIT = 6;

export const qualityCanvas = document.createElement("canvas");
export const qualityCtx = qualityCanvas.getContext("2d");
export const guidanceCanvas = document.createElement("canvas");
export const guidanceCtx = guidanceCanvas.getContext("2d");

export function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

export function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

export function buildExtractionPayload() {
  return {
    document_id: state.documentId,
    input_mode: "frontend",
    enable_correction: false,
    use_face_hint: false,
  };
}

export function getRenderRotation() {
  return state.rotation + state.fineRotation;
}

export function getNormalizedRotation() {
  return ((getRenderRotation() % 360) + 360) % 360;
}
