const state = {
  upload: null,
  documentId: null,
  previewImage: null,
  scale: 1,
  rotation: 0,
  offsetX: 0,
  offsetY: 0,
  dragPointer: null,
  dragOrigin: null,
  viewBaseScale: 1,
  viewSize: { width: 720, height: 480 },
  isBusy: false,
  extractionResult: null,
  animationFrameId: null,
  animationStartMs: 0,
};

const els = {
  uploadForm: document.querySelector("#upload-form"),
  fileInput: document.querySelector("#file-input"),
  uploadButton: document.querySelector("#upload-button"),
  rotateLeft: document.querySelector("#rotate-left"),
  rotateRight: document.querySelector("#rotate-right"),
  resetAdjust: document.querySelector("#reset-adjust"),
  extractButton: document.querySelector("#extract-button"),
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

const viewCtx = els.canvas.getContext("2d");
const exportCtx = els.exportCanvas.getContext("2d");

const VIEW_MIN_WIDTH = 320;
const VIEW_MIN_HEIGHT = 240;
const ZOOM_MIN = 0.85;
const ZOOM_MAX = 2.4;
const MRZ_GUIDE_HEIGHT_RATIO = 0.23;
const MRZ_GUIDE_WIDTH_RATIO = 0.82;
const MRZ_GUIDE_BOTTOM_MARGIN_RATIO = 0.08;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function setStatus(message, isError = false) {
  els.statusText.textContent = message;
  els.statusText.classList.toggle("status-error", isError);
}

function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function buildExtractionPayload() {
  return {
    document_id: state.documentId,
    input_mode: "frontend",
    enable_correction: false,
    use_face_hint: false,
  };
}

function updatePayloadView() {
  els.requestJson.textContent = formatJson(buildExtractionPayload());
}

function updateControls() {
  const hasDocument = Boolean(state.previewImage);
  els.rotateLeft.disabled = !hasDocument || state.isBusy;
  els.rotateRight.disabled = !hasDocument || state.isBusy;
  els.resetAdjust.disabled = !hasDocument || state.isBusy;
  els.extractButton.disabled = !hasDocument || state.isBusy;
  els.uploadButton.disabled = state.isBusy;
  els.fileInput.disabled = state.isBusy;
  els.zoomOut.disabled = !hasDocument || state.isBusy;
  els.zoomIn.disabled = !hasDocument || state.isBusy;
  els.zoomRange.disabled = !hasDocument || state.isBusy;
  els.rotationChip.textContent = `rotation: ${state.rotation}`;
  els.docIdChip.textContent = `document: ${state.documentId || "-"}`;
  els.engineChip.textContent = "render: dual canvas";
  els.zoomRange.value = String(state.scale);
  updatePayloadView();
}

function updateDocumentSummary() {
  if (!state.upload) {
    els.docName.textContent = "No document";
    els.docMeta.textContent = "Choose a file to begin. Geometry stays local until extraction.";
    return;
  }
  els.docName.textContent = state.upload.filename;
  els.docMeta.textContent = `${state.upload.source_type} | ${state.upload.preview_width}x${state.upload.preview_height}`;
}

function resetAdjustments() {
  state.scale = 1;
  state.rotation = 0;
  state.offsetX = 0;
  state.offsetY = 0;
  state.dragPointer = null;
  state.dragOrigin = null;
  drawViewCanvas();
  renderCropAnalysis();
  updateControls();
}

function resetViewAdjustments() {
  state.scale = 1;
  state.offsetX = 0;
  state.offsetY = 0;
  state.dragPointer = null;
  state.dragOrigin = null;
}

function loadImage(file) {
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

function getViewSize() {
  const width = Math.max(VIEW_MIN_WIDTH, Math.round(els.viewerFrame.clientWidth || VIEW_MIN_WIDTH));
  const height = Math.max(VIEW_MIN_HEIGHT, Math.round(els.viewerFrame.clientHeight || VIEW_MIN_HEIGHT));
  return { width, height };
}

function getNormalizedRotation() {
  return ((state.rotation % 360) + 360) % 360;
}

function getImageDisplaySize() {
  if (!state.previewImage) {
    return { width: 0, height: 0 };
  }
  const rotation = getNormalizedRotation();
  if (rotation === 90 || rotation === 270) {
    return { width: state.previewImage.height, height: state.previewImage.width };
  }
  return { width: state.previewImage.width, height: state.previewImage.height };
}

function computeViewBaseScale() {
  const imageSize = getImageDisplaySize();
  if (!imageSize.width || !imageSize.height) {
    return 1;
  }
  const viewSize = getViewSize();
  return Math.min(viewSize.width / imageSize.width, viewSize.height / imageSize.height);
}

function renderImage(targetCtx, canvas, baseScale, backgroundFill) {
  const image = state.previewImage;
  if (!image) {
    return;
  }

  targetCtx.clearRect(0, 0, canvas.width, canvas.height);
  targetCtx.fillStyle = backgroundFill;
  targetCtx.fillRect(0, 0, canvas.width, canvas.height);
  targetCtx.imageSmoothingEnabled = true;
  targetCtx.imageSmoothingQuality = "high";
  targetCtx.save();
  targetCtx.translate(canvas.width / 2, canvas.height / 2);
  targetCtx.rotate((state.rotation * Math.PI) / 180);
  targetCtx.scale(baseScale * state.scale, baseScale * state.scale);
  targetCtx.drawImage(
    image,
    -image.width / 2 + state.offsetX,
    -image.height / 2 + state.offsetY,
    image.width,
    image.height,
  );
  targetCtx.restore();
}

function computeMrzGuideRect(width = state.viewSize.width, height = state.viewSize.height) {
  const rectWidth = width * MRZ_GUIDE_WIDTH_RATIO;
  const rectHeight = height * MRZ_GUIDE_HEIGHT_RATIO;
  const x = (width - rectWidth) / 2;
  const y = height - rectHeight - (height * MRZ_GUIDE_BOTTOM_MARGIN_RATIO);
  return { x, y, width: rectWidth, height: rectHeight };
}

function getTransformedImageBounds() {
  if (!state.previewImage) {
    return null;
  }

  const image = state.previewImage;
  const scale = state.viewBaseScale * state.scale;
  const radians = (state.rotation * Math.PI) / 180;
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  const cx = state.viewSize.width / 2;
  const cy = state.viewSize.height / 2;
  const corners = [
    [-image.width / 2 + state.offsetX, -image.height / 2 + state.offsetY],
    [image.width / 2 + state.offsetX, -image.height / 2 + state.offsetY],
    [image.width / 2 + state.offsetX, image.height / 2 + state.offsetY],
    [-image.width / 2 + state.offsetX, image.height / 2 + state.offsetY],
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

function drawMrzOverlay() {
  const rect = computeMrzGuideRect();
  const labelY = Math.max(22, rect.y - 14);

  viewCtx.save();
  viewCtx.fillStyle = "rgba(8, 12, 18, 0.28)";
  viewCtx.beginPath();
  viewCtx.rect(0, 0, state.viewSize.width, state.viewSize.height);
  viewCtx.rect(rect.x, rect.y, rect.width, rect.height);
  viewCtx.fill("evenodd");

  viewCtx.strokeStyle = "rgba(88, 224, 255, 0.72)";
  viewCtx.lineWidth = 1.5;
  viewCtx.strokeRect(rect.x, rect.y, rect.width, rect.height);

  viewCtx.fillStyle = "rgba(196, 244, 255, 0.88)";
  viewCtx.font = '600 13px "Segoe UI", sans-serif';
  viewCtx.textAlign = "left";
  viewCtx.fillText("Align MRZ here", rect.x, labelY);
  viewCtx.restore();
}

function drawScanAnimation(timestampMs) {
  if (!state.previewImage) {
    return;
  }
  if (!state.animationStartMs) {
    state.animationStartMs = timestampMs;
  }

  const rect = computeMrzGuideRect();
  const elapsed = (timestampMs - state.animationStartMs) / 1000;
  const wave = (Math.sin(elapsed * 1.4) + 1) / 2;
  const y = rect.y + (wave * rect.height);

  viewCtx.save();
  viewCtx.beginPath();
  viewCtx.rect(rect.x, rect.y, rect.width, rect.height);
  viewCtx.clip();
  const gradient = viewCtx.createLinearGradient(rect.x, y - 8, rect.x, y + 8);
  gradient.addColorStop(0, "rgba(88, 224, 255, 0)");
  gradient.addColorStop(0.5, "rgba(88, 224, 255, 0.6)");
  gradient.addColorStop(1, "rgba(88, 224, 255, 0)");
  viewCtx.strokeStyle = gradient;
  viewCtx.lineWidth = 2;
  viewCtx.beginPath();
  viewCtx.moveTo(rect.x + 10, y);
  viewCtx.lineTo(rect.x + rect.width - 10, y);
  viewCtx.stroke();
  viewCtx.restore();
}

function getGuideStatus() {
  if (!state.previewImage) {
    return "No document loaded.";
  }

  const guide = computeMrzGuideRect();
  const bounds = getTransformedImageBounds();
  if (!bounds) {
    return "MRZ guide active";
  }

  const overlapsGuide =
    bounds.left < guide.x + guide.width &&
    bounds.right > guide.x &&
    bounds.top < guide.y + guide.height &&
    bounds.bottom > guide.y;

  if (!overlapsGuide) {
    return "Adjust position for optimal capture";
  }
  if (state.scale < 0.95) {
    return "Zoom in slightly for a stronger MRZ capture";
  }
  return "Ready for extraction";
}

function drawViewCanvas(timestampMs = performance.now()) {
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

function drawExportCanvas() {
  if (!state.previewImage) {
    return;
  }
  els.exportCanvas.width = state.previewImage.width;
  els.exportCanvas.height = state.previewImage.height;
  renderImage(exportCtx, els.exportCanvas, 1, "#ffffff");
}

function dataUrlToBlob(dataUrl) {
  const [meta, base64] = dataUrl.split(",");
  const mime = meta.match(/data:(.*?);base64/)?.[1] || "image/jpeg";
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mime });
}

function extractImage() {
  drawExportCanvas();
  return els.exportCanvas.toDataURL("image/jpeg", 0.95);
}

function buildAnalysis(result) {
  if (!result) {
    return [
      {
        title: "No extraction",
        items: ["Run extraction to see a structured summary."],
        tone: "",
      },
    ];
  }

  const parsed = result.parsed || {};
  const line1 = result.line1 || "";
  const line2 = result.line2 || "";
  const cards = [];
  const mrzItems = [line1 || "-", line2 || "-"];
  const structuralItems = [
    `Status: ${result.status || "unknown"}`,
    `Line 1 length: ${line1.length}/44`,
    `Line 2 length: ${line2.length}/44`,
  ];
  const identityItems = [];
  const documentItems = [];
  const warningItems = [];

  if (typeof result.duration_ms === "number") {
    structuralItems.push(`Duration: ${result.duration_ms.toFixed(2)} ms`);
  }
  if (result.report_path) {
    structuralItems.push({
      type: "link",
      label: "Report",
      href: `/api/extractions/${encodeURIComponent(result.extraction_id)}/report`,
      text: "Open JSON report",
    });
  }

  if (parsed.surname || parsed.given_names) {
    identityItems.push(`Surname: ${parsed.surname || "-"}`);
    identityItems.push(`Given names: ${parsed.given_names || "-"}`);
  }
  if (parsed.sex) {
    identityItems.push(`Sex: ${parsed.sex}`);
  }
  if (parsed.document_number) {
    documentItems.push(`Document number: ${parsed.document_number}`);
  }
  if (parsed.nationality) {
    documentItems.push(`Nationality: ${parsed.nationality}`);
  }
  if (parsed.birth_date_yymmdd) {
    documentItems.push(`Birth date: ${parsed.birth_date_yymmdd}`);
  }
  if (parsed.expiry_date_yymmdd) {
    documentItems.push(`Expiry date: ${parsed.expiry_date_yymmdd}`);
  }

  if (line1.length !== 44) {
    warningItems.push("Line 1 is not 44 characters, so TD3 structure is incomplete.");
  }
  if (line2.length !== 44) {
    warningItems.push("Line 2 is not 44 characters, so checksum-backed fields may be unreliable.");
  }
  if (!parsed.document_number) {
    warningItems.push("Document number was not parsed from line 2.");
  }
  if (!parsed.surname && !parsed.given_names) {
    warningItems.push("Name fields were not parsed from line 1.");
  }

  cards.push({ title: "MRZ Output", items: mrzItems, tone: "" });
  if (identityItems.length > 0) {
    cards.push({ title: "Identity Fields", items: identityItems, tone: "" });
  }
  if (documentItems.length > 0) {
    cards.push({ title: "Document Fields", items: documentItems, tone: "" });
  }
  cards.push({
    title: "Structural Summary",
    items: structuralItems,
    tone: warningItems.length === 0 ? "analysis-good" : "",
  });
  cards.push({
    title: "Quality Notes",
    items: warningItems.length > 0 ? warningItems : ["Line lengths and parsed fields look structurally plausible for TD3 output."],
    tone: warningItems.length > 0 ? "analysis-warn" : "analysis-good",
  });
  return cards;
}

function renderAnalysis(result) {
  const cards = buildAnalysis(result);
  els.analysisOutput.innerHTML = cards.map((card) => {
    const items = card.items.map((item) => {
      if (typeof item === "string") {
        const itemClass = card.title === "MRZ Output" ? "mrz-line" : "";
        return `<li class="${itemClass}">${escapeHtml(item)}</li>`;
      }
      if (item && item.type === "link") {
        return `<li>${escapeHtml(item.label)}: <a class="analysis-link" href="${escapeHtml(item.href)}" target="_blank" rel="noreferrer">${escapeHtml(item.text)}</a></li>`;
      }
      return "";
    }).join("");
    const listClass = card.title === "MRZ Output" ? "mrz-list" : "";
    return `
      <section class="analysis-card">
        <h3 class="${card.tone || ""}">${escapeHtml(card.title)}</h3>
        <ul class="${listClass}">${items}</ul>
      </section>
    `;
  }).join("");
}

function buildAdjustmentCards() {
  if (!state.previewImage) {
    return [
      {
        title: "No document",
        items: ["Upload a document before checking alignment."],
        tone: "",
      },
    ];
  }

  return [
    {
      title: "Adjustment Geometry",
      items: [
        `Rotation: ${state.rotation} degrees`,
        `Scale: ${state.scale.toFixed(2)}x`,
        `Offset: x=${state.offsetX.toFixed(1)}, y=${state.offsetY.toFixed(1)}`,
      ],
      tone: "analysis-good",
    },
    {
      title: "Guide Status",
      items: [
        "MRZ guide active",
        getGuideStatus(),
      ],
      tone: "analysis-good",
    },
  ];
}

function renderCropAnalysis() {
  const cards = buildAdjustmentCards();
  els.cropAnalysisOutput.innerHTML = cards.map((card) => {
    const items = card.items.map((item) => `<li>${escapeHtml(item)}</li>`).join("");
    return `
      <section class="analysis-card">
        <h3 class="${card.tone || ""}">${escapeHtml(card.title)}</h3>
        <ul>${items}</ul>
      </section>
    `;
  }).join("");
}

function animate(timestampMs) {
  drawViewCanvas(timestampMs);
  state.animationFrameId = window.requestAnimationFrame(animate);
}

function ensureAnimationLoop() {
  if (state.animationFrameId !== null) {
    return;
  }
  state.animationFrameId = window.requestAnimationFrame(animate);
}

function stopAnimationLoop() {
  if (state.animationFrameId === null) {
    return;
  }
  window.cancelAnimationFrame(state.animationFrameId);
  state.animationFrameId = null;
}

function getCanvasPointer(event) {
  const rect = els.canvas.getBoundingClientRect();
  return {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
}

function handlePointerDown(event) {
  if (!state.previewImage || state.isBusy) {
    return;
  }
  els.canvas.setPointerCapture(event.pointerId);
  state.dragPointer = event.pointerId;
  state.dragOrigin = getCanvasPointer(event);
}

function handlePointerMove(event) {
  if (state.dragPointer !== event.pointerId || !state.dragOrigin || !state.previewImage) {
    return;
  }
  const next = getCanvasPointer(event);
  const dx = next.x - state.dragOrigin.x;
  const dy = next.y - state.dragOrigin.y;
  const renderScale = state.viewBaseScale * state.scale;
  if (renderScale > 0) {
    state.offsetX += dx / renderScale;
    state.offsetY += dy / renderScale;
  }
  state.dragOrigin = next;
  renderCropAnalysis();
  setStatus(getGuideStatus());
}

function handlePointerUp(event) {
  if (state.dragPointer !== event.pointerId) {
    return;
  }
  els.canvas.releasePointerCapture(event.pointerId);
  state.dragPointer = null;
  state.dragOrigin = null;
  renderCropAnalysis();
  setStatus(getGuideStatus());
}

async function handleUpload(event) {
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
    els.uploadJson.textContent = formatJson(state.upload);
    els.resultJson.textContent = "No extraction yet.";
    renderAnalysis(null);
    renderCropAnalysis();
    setStatus("MRZ guide active");
    ensureAnimationLoop();
  } catch (error) {
    setStatus(error.message || "Local load failed.", true);
  } finally {
    state.isBusy = false;
    updateControls();
  }
}

function adjustZoom(nextScale) {
  state.scale = Number(clamp(nextScale, ZOOM_MIN, ZOOM_MAX).toFixed(2));
  drawViewCanvas();
  renderCropAnalysis();
  setStatus(getGuideStatus());
  updateControls();
}

function rotate(delta) {
  if (!state.previewImage) {
    return;
  }
  state.rotation = (state.rotation + delta + 360) % 360;
  resetViewAdjustments();
  drawViewCanvas();
  setStatus(getGuideStatus());
  renderCropAnalysis();
  updateControls();
}

async function handleExtraction() {
  if (!state.previewImage) {
    return;
  }

  state.isBusy = true;
  updateControls();
  setStatus("Rendering full-resolution export ...");

  try {
    const imageDataUrl = extractImage();
    const blob = dataUrlToBlob(imageDataUrl);
    const uploadName = `${(state.upload?.filename || "document").replace(/\.[^.]+$/, "")}_frontend.jpg`;
    const formData = new FormData();
    formData.append("file", blob, uploadName);

    setStatus("Uploading final prepared image ...");
    const uploadResponse = await fetch("/api/uploads", {
      method: "POST",
      body: formData,
    });
    const uploadPayload = await uploadResponse.json();
    if (!uploadResponse.ok) {
      throw new Error(uploadPayload.detail || "Frontend image upload failed.");
    }

    state.documentId = uploadPayload.document_id;
    els.uploadJson.textContent = formatJson(uploadPayload);
    updateControls();

    const payload = buildExtractionPayload();
    setStatus("Running extraction ...");
    const response = await fetch("/api/extractions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Extraction failed.");
    }

    state.extractionResult = result;
    els.resultJson.textContent = formatJson(result);
    renderAnalysis(result);
    setStatus(`Extraction completed with status=${result.status}.`);
  } catch (error) {
    setStatus(error.message || "Extraction failed.", true);
  } finally {
    state.isBusy = false;
    updateControls();
  }
}

function handleResize() {
  drawViewCanvas();
  renderCropAnalysis();
}

function init() {
  els.exportCanvas.hidden = true;
  updateDocumentSummary();
  updateControls();
  renderAnalysis(null);
  renderCropAnalysis();
  drawViewCanvas();

  els.uploadForm.addEventListener("submit", handleUpload);
  els.rotateLeft.addEventListener("click", () => rotate(-90));
  els.rotateRight.addEventListener("click", () => rotate(90));
  els.resetAdjust.addEventListener("click", resetAdjustments);
  els.extractButton.addEventListener("click", handleExtraction);
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

init();
