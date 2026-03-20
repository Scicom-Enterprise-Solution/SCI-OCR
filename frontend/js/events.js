import { renderAnalysis, renderCropAnalysis } from "./analysis.js";
import { setStatus, updateControls, updateDocumentSummary, els } from "./dom.js";
import { saveExportImage, handleExtraction } from "./extract.js";
import { getGuideStatus } from "./guidance.js";
import { drawViewCanvas, getExportCanvasSize } from "./render.js";
import { ROTATE_DRAG_SENSITIVITY, ZOOM_MAX, ZOOM_MIN, clamp, getRenderRotation, state } from "./state.js";

export function resetAdjustments() {
  state.scale = 1;
  state.rotation = 0;
  state.fineRotation = 0;
  state.offsetX = 0;
  state.offsetY = 0;
  state.dragPointer = null;
  state.dragOrigin = null;
  state.dragMode = null;
  state.guidance.mrzRect = null;
  state.guidance.mrzScore = 0;
  state.guidance.faceRect = null;
  state.guidance.faceConfidence = 0;
  drawViewCanvas();
  renderCropAnalysis();
  updateControls();
}

export function resetViewAdjustments() {
  state.scale = 1;
  state.offsetX = 0;
  state.offsetY = 0;
  state.dragPointer = null;
  state.dragOrigin = null;
  state.dragMode = null;
}

export function loadImage(file) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    const objectUrl = URL.createObjectURL(file);
    image.onload = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(image);
    };
    image.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Failed to load the selected image."));
    };
    image.src = objectUrl;
  });
}

export function getCanvasPointer(event) {
  const rect = els.canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

export function handlePointerDown(event) {
  if (!state.previewImage || state.isBusy) {
    return;
  }
  els.canvas.setPointerCapture(event.pointerId);
  state.dragPointer = event.pointerId;
  state.dragOrigin = getCanvasPointer(event);
  state.dragMode = event.altKey ? "rotate" : "pan";
}

export function handlePointerMove(event) {
  if (state.dragPointer !== event.pointerId || !state.dragOrigin || !state.previewImage) {
    return;
  }
  const next = getCanvasPointer(event);
  const dx = next.x - state.dragOrigin.x;
  const dy = next.y - state.dragOrigin.y;
  if (state.dragMode === "rotate") {
    state.fineRotation = Number(clamp(state.fineRotation + (dx * ROTATE_DRAG_SENSITIVITY), -12, 12).toFixed(1));
    state.dragOrigin = next;
    drawViewCanvas();
    renderCropAnalysis();
    updateControls();
    setStatus(getGuideStatus());
    return;
  }
  const effectiveScale = state.viewBaseScale * state.scale;
  const radians = (getRenderRotation() * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  if (effectiveScale > 0 && state.previewImage) {
    const localDx = ((dx * cos) + (dy * sin)) / effectiveScale;
    const localDy = ((-dx * sin) + (dy * cos)) / effectiveScale;
    state.offsetX += localDx / state.previewImage.width;
    state.offsetY += localDy / state.previewImage.height;
  }
  state.dragOrigin = next;
  drawViewCanvas();
  renderCropAnalysis();
  setStatus(getGuideStatus());
}

export function handlePointerUp(event) {
  if (state.dragPointer !== event.pointerId) {
    return;
  }
  els.canvas.releasePointerCapture(event.pointerId);
  state.dragPointer = null;
  state.dragOrigin = null;
  state.dragMode = null;
  renderCropAnalysis();
  setStatus(getGuideStatus());
}

export async function handleUpload(event) {
  event.preventDefault();
  const file = els.fileInput.files?.[0];
  if (!file) {
    setStatus("Choose a file before loading.", true);
    return;
  }

  state.isBusy = true;
  updateControls();
  setStatus(`Loading ${file.name} locally ...`);

  try {
    const image = await loadImage(file);
    state.upload = {
      filename: file.name,
      source_type: "local_image",
      extension: file.name.includes(".") ? file.name.slice(file.name.lastIndexOf(".")).toLowerCase() : "",
      preview_width: image.width,
      preview_height: image.height,
    };
    state.previewImage = image;
    state.documentId = null;
    state.extractionResult = null;
    state.animationStartMs = 0;
    resetAdjustments();
    updateDocumentSummary();
    els.uploadJson.textContent = JSON.stringify(state.upload, null, 2);
    els.resultJson.textContent = "No extraction yet.";
    renderAnalysis(null);
    renderCropAnalysis();
    setStatus("MRZ guide active");
  } catch (error) {
    setStatus(error.message || "Local load failed.", true);
  } finally {
    state.isBusy = false;
    updateControls();
  }
}

export function adjustZoom(nextScale) {
  state.scale = Number(clamp(nextScale, ZOOM_MIN, ZOOM_MAX).toFixed(2));
  drawViewCanvas();
  renderCropAnalysis();
  setStatus(getGuideStatus());
  updateControls();
}

export function rotate(delta) {
  if (!state.previewImage) {
    return;
  }
  state.rotation = Number((state.rotation + delta).toFixed(1));
  state.fineRotation = 0;
  resetViewAdjustments();
  drawViewCanvas();
  setStatus(getGuideStatus());
  renderCropAnalysis();
  updateControls();
}

export function handleResize() {
  drawViewCanvas();
  renderCropAnalysis();
}

export function attachEventListeners() {
  els.uploadForm.addEventListener("submit", handleUpload);
  els.rotateLeft.addEventListener("click", () => rotate(-90));
  els.rotateRight.addEventListener("click", () => rotate(90));
  els.resetAdjust.addEventListener("click", resetAdjustments);
  els.saveExportButton.addEventListener("click", saveExportImage);
  els.extractButton.addEventListener("click", handleExtraction);
  els.angleRange.addEventListener("input", () => {
    state.fineRotation = Number(els.angleRange.value);
    drawViewCanvas();
    renderCropAnalysis();
    updateControls();
    setStatus(getGuideStatus());
  });
  els.zoomRange.addEventListener("input", () => adjustZoom(Number(els.zoomRange.value)));
  els.zoomOut.addEventListener("click", () => adjustZoom(state.scale - 0.05));
  els.zoomIn.addEventListener("click", () => adjustZoom(state.scale + 0.05));
  els.canvas.addEventListener("pointerdown", handlePointerDown);
  els.canvas.addEventListener("pointermove", handlePointerMove);
  els.canvas.addEventListener("pointerup", handlePointerUp);
  els.canvas.addEventListener("pointercancel", handlePointerUp);
  els.canvas.addEventListener("pointerleave", handlePointerUp);
  window.addEventListener("resize", handleResize);
}
