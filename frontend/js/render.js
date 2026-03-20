import {
  CAPTURE_ASPECT_RATIO,
  MRZ_GUIDE_BOTTOM_MARGIN_RATIO,
  MRZ_GUIDE_HEIGHT_RATIO,
  MRZ_GUIDE_WIDTH_RATIO,
  VIEW_MIN_HEIGHT,
  VIEW_MIN_WIDTH,
  getNormalizedRotation,
  getRenderRotation,
  state,
} from "./state.js";
import { els, exportCtx, viewCtx } from "./dom.js";

export function getViewSize() {
  const availableWidth = Math.max(VIEW_MIN_WIDTH, Math.round(els.viewerFrame.clientWidth || VIEW_MIN_WIDTH));
  const availableHeight = Math.max(VIEW_MIN_HEIGHT, Math.round(els.viewerFrame.clientHeight || VIEW_MIN_HEIGHT));
  let width = availableWidth;
  let height = Math.round(width / CAPTURE_ASPECT_RATIO);

  if (height > availableHeight) {
    height = availableHeight;
    width = Math.round(height * CAPTURE_ASPECT_RATIO);
  }

  return { width, height };
}

export function getImageDisplaySize() {
  if (!state.previewImage) {
    return { width: 0, height: 0 };
  }
  const radians = (getNormalizedRotation() * Math.PI) / 180;
  const cos = Math.abs(Math.cos(radians));
  const sin = Math.abs(Math.sin(radians));
  return {
    width: (state.previewImage.width * cos) + (state.previewImage.height * sin),
    height: (state.previewImage.width * sin) + (state.previewImage.height * cos),
  };
}

export function computeViewBaseScale() {
  const imageSize = getImageDisplaySize();
  if (!imageSize.width || !imageSize.height) {
    return 1;
  }
  const viewSize = getViewSize();
  return Math.min(viewSize.width / imageSize.width, viewSize.height / imageSize.height);
}

export function getImageOffsetPixels() {
  if (!state.previewImage) {
    return { x: 0, y: 0 };
  }
  return {
    x: state.offsetX * state.previewImage.width,
    y: state.offsetY * state.previewImage.height,
  };
}

export function getExportCanvasSize() {
  if (!state.previewImage) {
    return { width: 0, height: 0 };
  }
  const viewAspect = state.viewSize.width / state.viewSize.height;
  const sourceAspect = state.previewImage.width / state.previewImage.height;

  if (sourceAspect >= viewAspect) {
    return {
      width: Math.round(state.previewImage.height * viewAspect),
      height: state.previewImage.height,
    };
  }
  return {
    width: state.previewImage.width,
    height: Math.round(state.previewImage.width / viewAspect),
  };
}

export function renderImage(targetCtx, canvas, baseScale, backgroundFill) {
  const image = state.previewImage;
  if (!image) {
    return;
  }
  const offsetPixels = getImageOffsetPixels();

  targetCtx.clearRect(0, 0, canvas.width, canvas.height);
  targetCtx.fillStyle = backgroundFill;
  targetCtx.fillRect(0, 0, canvas.width, canvas.height);
  targetCtx.imageSmoothingEnabled = true;
  targetCtx.imageSmoothingQuality = "high";
  targetCtx.save();
  targetCtx.translate(canvas.width / 2, canvas.height / 2);
  targetCtx.rotate((getRenderRotation() * Math.PI) / 180);
  targetCtx.scale(baseScale * state.scale, baseScale * state.scale);
  targetCtx.drawImage(
    image,
    -image.width / 2 + offsetPixels.x,
    -image.height / 2 + offsetPixels.y,
    image.width,
    image.height,
  );
  targetCtx.restore();
}

export function renderGuidanceSource(targetCanvas, targetCtx) {
  targetCanvas.width = state.viewSize.width;
  targetCanvas.height = state.viewSize.height;
  renderImage(targetCtx, targetCanvas, state.viewBaseScale, "#101417");
}

export function computeMrzGuideRect(width = state.viewSize.width, height = state.viewSize.height) {
  const rectWidth = width * MRZ_GUIDE_WIDTH_RATIO;
  const rectHeight = height * MRZ_GUIDE_HEIGHT_RATIO;
  const x = (width - rectWidth) / 2;
  const y = height - rectHeight - (height * MRZ_GUIDE_BOTTOM_MARGIN_RATIO);
  return { x, y, width: rectWidth, height: rectHeight };
}

export function getTransformedImageBounds() {
  if (!state.previewImage) {
    return null;
  }

  const image = state.previewImage;
  const scale = state.viewBaseScale * state.scale;
  const offsetPixels = getImageOffsetPixels();
  const radians = (getRenderRotation() * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  const cx = state.viewSize.width / 2;
  const cy = state.viewSize.height / 2;
  const corners = [
    [-image.width / 2 + offsetPixels.x, -image.height / 2 + offsetPixels.y],
    [image.width / 2 + offsetPixels.x, -image.height / 2 + offsetPixels.y],
    [image.width / 2 + offsetPixels.x, image.height / 2 + offsetPixels.y],
    [-image.width / 2 + offsetPixels.x, image.height / 2 + offsetPixels.y],
  ].map(([x, y]) => ({
    x: cx + ((x * cos) - (y * sin)) * scale,
    y: cy + ((x * sin) + (y * cos)) * scale,
  }));

  const xs = corners.map((point) => point.x);
  const ys = corners.map((point) => point.y);
  return {
    left: Math.min(...xs),
    right: Math.max(...xs),
    top: Math.min(...ys),
    bottom: Math.max(...ys),
  };
}

export function drawMrzOverlay() {
  const rect = computeMrzGuideRect();
  const labelY = Math.max(22, rect.y - 14);
  const guideLocked = Boolean(state.guidance.mrzRect);

  viewCtx.save();
  viewCtx.fillStyle = "rgba(8, 12, 18, 0.28)";
  viewCtx.beginPath();
  viewCtx.rect(0, 0, state.viewSize.width, state.viewSize.height);
  viewCtx.rect(rect.x, rect.y, rect.width, rect.height);
  viewCtx.fill("evenodd");

  viewCtx.strokeStyle = guideLocked ? "rgba(66, 224, 120, 0.9)" : "rgba(255, 90, 90, 0.78)";
  viewCtx.lineWidth = guideLocked ? 1.8 : 1.4;
  viewCtx.shadowColor = guideLocked ? "rgba(66, 224, 120, 0.4)" : "rgba(255, 96, 96, 0.38)";
  viewCtx.shadowBlur = 12;
  viewCtx.strokeRect(rect.x, rect.y, rect.width, rect.height);
  viewCtx.shadowBlur = 0;

  viewCtx.fillStyle = guideLocked ? "rgba(216, 255, 224, 0.94)" : "rgba(255, 228, 228, 0.92)";
  viewCtx.font = '600 13px "Segoe UI", sans-serif';
  viewCtx.textAlign = "left";
  viewCtx.fillText(guideLocked ? "MRZ locked" : "Align MRZ here", rect.x, labelY);

  if (state.guidance.mrzRect) {
    const mrz = state.guidance.mrzRect;
    viewCtx.strokeStyle = "rgba(66, 224, 120, 0.92)";
    viewCtx.lineWidth = 1.6;
    viewCtx.setLineDash([5, 4]);
    viewCtx.strokeRect(mrz.x, mrz.y, mrz.width, mrz.height);
    viewCtx.setLineDash([]);
  }

  if (state.guidance.faceRect) {
    const face = state.guidance.faceRect;
    viewCtx.strokeStyle = "rgba(255, 204, 84, 0.9)";
    viewCtx.lineWidth = 1.5;
    viewCtx.strokeRect(face.x, face.y, face.width, face.height);
  }
  viewCtx.restore();
}

export function drawScanAnimation(timestampMs) {
  if (!state.previewImage) {
    return;
  }
  if (!state.animationStartMs) {
    state.animationStartMs = timestampMs;
  }

  const rect = computeMrzGuideRect();
  const elapsed = (timestampMs - state.animationStartMs) / 1000;
  const wave = (Math.sin(elapsed * 1.1) + 1) / 2;
  const y = rect.y + (wave * rect.height);

  viewCtx.save();
  viewCtx.beginPath();
  viewCtx.rect(rect.x, rect.y, rect.width, rect.height);
  viewCtx.clip();
  const beamColor = state.guidance.mrzRect ? "66, 224, 120" : "255, 92, 92";
  const glow = viewCtx.createLinearGradient(rect.x, y - 18, rect.x, y + 18);
  glow.addColorStop(0, `rgba(${beamColor}, 0)`);
  glow.addColorStop(0.35, `rgba(${beamColor}, 0.18)`);
  glow.addColorStop(0.5, `rgba(${beamColor}, 0.4)`);
  glow.addColorStop(0.65, `rgba(${beamColor}, 0.18)`);
  glow.addColorStop(1, `rgba(${beamColor}, 0)`);
  viewCtx.fillStyle = glow;
  viewCtx.fillRect(rect.x, y - 18, rect.width, 36);

  const lineGradient = viewCtx.createLinearGradient(rect.x, y - 3, rect.x, y + 3);
  lineGradient.addColorStop(0, `rgba(${beamColor}, 0)`);
  lineGradient.addColorStop(0.5, `rgba(${beamColor}, 0.95)`);
  lineGradient.addColorStop(1, `rgba(${beamColor}, 0)`);
  viewCtx.strokeStyle = lineGradient;
  viewCtx.lineWidth = 2.2;
  viewCtx.shadowColor = `rgba(${beamColor}, 0.65)`;
  viewCtx.shadowBlur = 10;
  viewCtx.beginPath();
  viewCtx.moveTo(rect.x + 10, y);
  viewCtx.lineTo(rect.x + rect.width - 10, y);
  viewCtx.stroke();
  viewCtx.shadowBlur = 0;
  viewCtx.restore();
}

export function drawViewCanvas(timestampMs = performance.now()) {
  const viewSize = getViewSize();
  state.viewSize = viewSize;
  state.viewBaseScale = computeViewBaseScale();
  els.viewerFrame.classList.toggle("empty", !state.previewImage);

  els.canvas.width = viewSize.width;
  els.canvas.height = viewSize.height;

  if (!state.previewImage) {
    viewCtx.clearRect(0, 0, viewSize.width, viewSize.height);
    viewCtx.fillStyle = "#f6f0e5";
    viewCtx.fillRect(0, 0, viewSize.width, viewSize.height);
    viewCtx.fillStyle = "#67757c";
    viewCtx.font = '600 18px "Segoe UI", sans-serif';
    viewCtx.textAlign = "center";
    viewCtx.fillText("Upload a document to preview it here.", viewSize.width / 2, viewSize.height / 2);
    return;
  }

  renderImage(viewCtx, els.canvas, state.viewBaseScale, "#101417");
  drawMrzOverlay();
  drawScanAnimation(timestampMs);
}

export function drawExportCanvas() {
  if (!state.previewImage) {
    return;
  }
  const exportSize = getExportCanvasSize();
  els.exportCanvas.width = exportSize.width;
  els.exportCanvas.height = exportSize.height;

  const exportScale = Math.min(
    els.exportCanvas.width / state.viewSize.width,
    els.exportCanvas.height / state.viewSize.height,
  );
  renderImage(exportCtx, els.exportCanvas, state.viewBaseScale * exportScale, "#ffffff");
}
