import { renderAnalysis, renderCropAnalysis } from "./analysis.js";
import { setStatus, updateControls, updateDocumentSummary, els } from "./dom.js";
import { getGuideStatus, runGuidanceAnalysis, ensureCvReady } from "./guidance.js";
import { drawViewCanvas } from "./render.js";
import { state } from "./state.js";
import { attachEventListeners } from "./events.js";

function animate(timestampMs) {
  drawViewCanvas(timestampMs);
  runGuidanceAnalysis(timestampMs);
  renderCropAnalysis();
  if (!state.isBusy) {
    setStatus(getGuideStatus());
  }
  state.animationFrameId = window.requestAnimationFrame(animate);
}

function ensureAnimationLoop() {
  if (state.animationFrameId !== null) {
    return;
  }
  state.animationFrameId = window.requestAnimationFrame(animate);
}

function init() {
  els.exportCanvas.hidden = true;
  if (!ensureCvReady()) {
    window.setTimeout(() => {
      if (!ensureCvReady() && state.guidance.cvStatus !== "ready") {
        state.guidance.cvStatus = typeof window.cv === "undefined" ? "loading" : state.guidance.cvStatus;
        updateControls();
      }
    }, 500);
  }

  updateDocumentSummary();
  updateControls();
  renderAnalysis(null);
  renderCropAnalysis();
  drawViewCanvas();
  attachEventListeners();
  ensureAnimationLoop();
}

init();
