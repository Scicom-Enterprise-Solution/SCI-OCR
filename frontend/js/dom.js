import { buildExtractionPayload, formatJson, getRenderRotation, state } from "./state.js";

export const els = {
  uploadForm: document.querySelector("#upload-form"),
  fileInput: document.querySelector("#file-input"),
  uploadButton: document.querySelector("#upload-button"),
  rotateLeft: document.querySelector("#rotate-left"),
  rotateRight: document.querySelector("#rotate-right"),
  resetAdjust: document.querySelector("#reset-adjust"),
  saveExportButton: document.querySelector("#save-export"),
  extractButton: document.querySelector("#extract-button"),
  angleRange: document.querySelector("#angle-range"),
  zoomOut: document.querySelector("#zoom-out"),
  zoomIn: document.querySelector("#zoom-in"),
  zoomRange: document.querySelector("#zoom-range"),
  canvas: document.querySelector("#preview-canvas"),
  exportCanvas: document.querySelector("#export-canvas"),
  viewerFrame: document.querySelector("#viewer-frame"),
  docName: document.querySelector("#doc-name"),
  docMeta: document.querySelector("#doc-meta"),
  docIdChip: document.querySelector("#doc-id-chip"),
  rotationChip: document.querySelector("#rotation-chip"),
  engineChip: document.querySelector("#engine-chip"),
  cropAnalysisOutput: document.querySelector("#crop-analysis-output"),
  requestJson: document.querySelector("#request-json"),
  uploadJson: document.querySelector("#upload-json"),
  resultJson: document.querySelector("#result-json"),
  analysisOutput: document.querySelector("#analysis-output"),
  statusText: document.querySelector("#status-text"),
};

export const viewCtx = els.canvas.getContext("2d");
export const exportCtx = els.exportCanvas.getContext("2d");

export function setStatus(message, isError = false) {
  els.statusText.textContent = message;
  els.statusText.classList.toggle("status-error", isError);
}

export function updatePayloadView() {
  els.requestJson.textContent = formatJson(buildExtractionPayload());
}

export function updateControls() {
  const hasDocument = Boolean(state.previewImage);
  els.rotateLeft.disabled = !hasDocument || state.isBusy;
  els.rotateRight.disabled = !hasDocument || state.isBusy;
  els.resetAdjust.disabled = !hasDocument || state.isBusy;
  els.saveExportButton.disabled = !hasDocument || state.isBusy;
  els.extractButton.disabled = !hasDocument || state.isBusy;
  els.uploadButton.disabled = state.isBusy;
  els.fileInput.disabled = state.isBusy;
  els.angleRange.disabled = !hasDocument || state.isBusy;
  els.zoomOut.disabled = !hasDocument || state.isBusy;
  els.zoomIn.disabled = !hasDocument || state.isBusy;
  els.zoomRange.disabled = !hasDocument || state.isBusy;
  els.rotationChip.textContent = `rotation: ${getRenderRotation().toFixed(1)}`;
  els.docIdChip.textContent = `document: ${state.documentId || "-"}`;
  const cvLabel = state.guidance.cvStatus === "ready"
    ? "OpenCV.js ready"
    : state.guidance.cvStatus === "error"
      ? "OpenCV.js unavailable"
      : "OpenCV.js loading";
  els.engineChip.textContent = `render: dual canvas | ${cvLabel}`;
  els.angleRange.value = String(state.fineRotation);
  els.zoomRange.value = String(state.scale);
  updatePayloadView();
}

export function updateDocumentSummary() {
  if (!state.upload) {
    els.docName.textContent = "No document";
    els.docMeta.textContent = "Choose a file to begin. Geometry stays local until extraction.";
    return;
  }
  els.docName.textContent = state.upload.filename;
  els.docMeta.textContent = `${state.upload.source_type} | ${state.upload.preview_width}x${state.upload.preview_height}`;
}
