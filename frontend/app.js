const state = {
  upload: null,
  documentId: null,
  previewImage: null,
  rotation: 0,
  microRotation: 0,
  zoom: 1,
  offsetX: 0,
  offsetY: 0,
  dragStart: null,
  dragCurrent: null,
  canvasScale: 1,
  canvasBounds: { x: 0, y: 0, width: 0, height: 0 },
  isBusy: false,
  previewMat: null,
};

const els = {
  uploadForm: document.querySelector("#upload-form"),
  fileInput: document.querySelector("#file-input"),
  uploadButton: document.querySelector("#upload-button"),
  rotateLeft: document.querySelector("#rotate-left"),
  rotateRight: document.querySelector("#rotate-right"),
  resetAdjust: document.querySelector("#reset-adjust"),
  extractButton: document.querySelector("#extract-button"),
  useFaceHint: document.querySelector("#use-face-hint"),
  microRotate: document.querySelector("#micro-rotate"),
  zoomOut: document.querySelector("#zoom-out"),
  zoomIn: document.querySelector("#zoom-in"),
  zoomRange: document.querySelector("#zoom-range"),
  offsetXRange: document.querySelector("#offset-x-range"),
  offsetYRange: document.querySelector("#offset-y-range"),
  canvas: document.querySelector("#preview-canvas"),
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

const ctx = els.canvas.getContext("2d");
const WORKING_FRAME_MARGIN_X = 0.04;
const WORKING_FRAME_MARGIN_Y = 0.04;
const TRANSFORM_PAD_RATIO = 0.14;
const OFFSET_LIMIT = 0.2;
const OFFSET_Y_LIMIT = 0.6;
const ZOOM_MIN = 1.0;
const ZOOM_MAX = 2.2;
const DRAG_SENSITIVITY = 0.35;
const MRZ_FOCUS_HEIGHT = 0.30;
const cvState = {
  ready: false,
  failed: false,
  pollCount: 0,
  hooked: false,
};

function setStatus(message, isError = false) {
  els.statusText.textContent = message;
  els.statusText.classList.toggle("status-error", isError);
}

function updateEngineChip() {
  if (cvState.ready) {
    els.engineChip.textContent = "preview: OpenCV.js";
    els.engineChip.classList.add("chip-ready");
    els.engineChip.classList.remove("chip-muted");
    return;
  }
  if (cvState.failed) {
    els.engineChip.textContent = "preview: canvas only";
    els.engineChip.classList.add("chip-muted");
    els.engineChip.classList.remove("chip-ready");
    return;
  }
  els.engineChip.textContent = "preview: loading OpenCV.js";
  els.engineChip.classList.add("chip-muted");
  els.engineChip.classList.remove("chip-ready");
}

function markOpenCvReady() {
  if (cvState.ready) {
    return;
  }
  cvState.ready = true;
  cvState.failed = false;
  updateEngineChip();
  renderCanvas();
}

function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function getPreviewUrl(documentId) {
  return `/api/documents/${documentId}/preview`;
}

function getWorkingFrameCrop() {
  const width = state.canvasBounds.width;
  const height = state.canvasBounds.height;
  if (!width || !height) {
    return null;
  }

  return {
    x: Number(WORKING_FRAME_MARGIN_X.toFixed(6)),
    y: Number(WORKING_FRAME_MARGIN_Y.toFixed(6)),
    width: Number((1.0 - (WORKING_FRAME_MARGIN_X * 2)).toFixed(6)),
    height: Number((1.0 - (WORKING_FRAME_MARGIN_Y * 2)).toFixed(6)),
  };
}

function getRotatedImageSize() {
  if (!state.previewImage) {
    return { width: 0, height: 0 };
  }
  const swapAxes = state.rotation === 90 || state.rotation === 270;
  return swapAxes
    ? { width: state.previewImage.height, height: state.previewImage.width }
    : { width: state.previewImage.width, height: state.previewImage.height };
}

function releaseMat(key) {
  const mat = state[key];
  if (mat && typeof mat.delete === "function") {
    mat.delete();
  }
  state[key] = null;
}

function clearPreviewMats() {
  releaseMat("previewMat");
}

function getPreviewMat() {
  if (!state.previewImage || !window.cv || !cvState.ready) {
    return null;
  }
  if (state.previewMat) {
    return state.previewMat;
  }

  const canvas = document.createElement("canvas");
  canvas.width = state.previewImage.width;
  canvas.height = state.previewImage.height;
  const offscreen = canvas.getContext("2d");
  offscreen.drawImage(state.previewImage, 0, 0);
  state.previewMat = window.cv.imread(canvas);
  return state.previewMat;
}

function getPaddedPreviewSource() {
  if (!state.previewImage) {
    return null;
  }
  const image = state.previewImage;
  const padX = Math.round(image.width * TRANSFORM_PAD_RATIO);
  const padY = Math.round(image.height * TRANSFORM_PAD_RATIO);
  const canvas = document.createElement("canvas");
  canvas.width = image.width + (padX * 2);
  canvas.height = image.height + (padY * 2);
  const offscreen = canvas.getContext("2d");
  offscreen.imageSmoothingEnabled = true;
  offscreen.imageSmoothingQuality = "high";
  offscreen.drawImage(image, padX, padY);
  return canvas;
}

function buildExtractionPayload() {
  return {
    document_id: state.documentId,
    rotation: state.rotation,
    transform: {
      micro_rotation: Number(state.microRotation.toFixed(3)),
      zoom: Number(state.zoom.toFixed(3)),
      offset_x: Number(state.offsetX.toFixed(6)),
      offset_y: Number(state.offsetY.toFixed(6)),
      viewport_width: Math.max(1, Math.round(state.canvasBounds.width || 0)),
      viewport_height: Math.max(1, Math.round(state.canvasBounds.height || 0)),
    },
    crop: getWorkingFrameCrop(),
    use_face_hint: Boolean(els.useFaceHint.checked),
  };
}

function updatePayloadView() {
  els.requestJson.textContent = formatJson(buildExtractionPayload());
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function getReportFilename(reportPath) {
  if (!reportPath) {
    return "";
  }
  const normalized = String(reportPath).replaceAll("\\", "/");
  const parts = normalized.split("/");
  return parts[parts.length - 1] || normalized;
}

function formatDateYYMMDD(value) {
  if (!value || !/^\d{6}$/.test(value)) {
    return null;
  }
  return `${value.slice(0, 2)}-${value.slice(2, 4)}-${value.slice(4, 6)}`;
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
  const mrzItems = [];
  const structuralItems = [];
  const identityItems = [];
  const documentItems = [];
  const warningItems = [];

  mrzItems.push(line1 || "-");
  mrzItems.push(line2 || "-");

  structuralItems.push(`Status: ${result.status || "unknown"}`);
  structuralItems.push(`Line 1 length: ${line1.length}/44`);
  structuralItems.push(`Line 2 length: ${line2.length}/44`);
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
    documentItems.push(
      `Birth date: ${parsed.birth_date_yymmdd}${formatDateYYMMDD(parsed.birth_date_yymmdd) ? ` (${formatDateYYMMDD(parsed.birth_date_yymmdd)})` : ""}`
    );
  }
  if (parsed.expiry_date_yymmdd) {
    documentItems.push(
      `Expiry date: ${parsed.expiry_date_yymmdd}${formatDateYYMMDD(parsed.expiry_date_yymmdd) ? ` (${formatDateYYMMDD(parsed.expiry_date_yymmdd)})` : ""}`
    );
  }
  if (parsed.personal_number) {
    documentItems.push(`Personal number: ${parsed.personal_number}`);
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
  if (!parsed.nationality) {
    warningItems.push("Nationality was not parsed from line 2.");
  }
  if (!parsed.surname && !parsed.given_names) {
    warningItems.push("Name fields were not parsed from line 1.");
  }

  cards.push({
    title: "MRZ Output",
    items: mrzItems,
    tone: "",
  });

  if (identityItems.length > 0) {
    cards.push({
      title: "Identity Fields",
      items: identityItems,
      tone: "",
    });
  }

  if (documentItems.length > 0) {
    cards.push({
      title: "Document Fields",
      items: documentItems,
      tone: "",
    });
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
  els.analysisOutput.innerHTML = cards
    .map((card) => {
      const items = card.items
        .map((item) => {
          if (typeof item === "string") {
            const itemClass = card.title === "MRZ Output" ? "mrz-line" : "";
            return `<li class="${itemClass}">${escapeHtml(item)}</li>`;
          }
          if (item && item.type === "link") {
            return `<li>${escapeHtml(item.label)}: <a class="analysis-link" href="${escapeHtml(item.href)}" target="_blank" rel="noreferrer">${escapeHtml(item.text)}</a></li>`;
          }
          return "";
        })
        .join("");
      const listClass = card.title === "MRZ Output" ? "mrz-list" : "";
      return `
        <section class="analysis-card">
          <h3 class="${card.tone || ""}">${escapeHtml(card.title)}</h3>
          <ul class="${listClass}">${items}</ul>
        </section>
      `;
    })
    .join("");
}

function buildCropAnalysis() {
  if (!state.previewImage) {
    return [
      {
        title: "No document",
        items: ["Upload a document before checking alignment."],
        tone: "",
      },
    ];
  }
  const warnings = [];
  const fixedCrop = getWorkingFrameCrop();

  if (Math.abs(state.microRotation) > 3.0) {
    warnings.push("Micro rotation is fairly large. Recheck that the page is truly skewed before keeping it.");
  }
  if (state.zoom > 1.6) {
    warnings.push("Zoom is high. Make sure important document edges are not pushed out of frame.");
  }
  if (Math.abs(state.offsetX) > 0.12) {
    warnings.push("Horizontal shift is large. Verify the passport page is still centered enough for extraction.");
  }
  if (Math.abs(state.offsetY) > 0.35) {
    warnings.push("Vertical shift is large. Verify the passport page is still fully visible.");
  }

  const cards = [
    {
      title: "Adjustment Geometry",
      items: [
        `Rotation: ${state.rotation} degrees`,
        `Micro rotation: ${state.microRotation.toFixed(1)} degrees`,
        `Zoom: ${state.zoom.toFixed(2)}x`,
        `Offset: x=${state.offsetX.toFixed(3)}, y=${state.offsetY.toFixed(3)}`,
        `Fixed crop: x=${fixedCrop.x.toFixed(3)}, y=${fixedCrop.y.toFixed(3)}, width=${fixedCrop.width.toFixed(3)}, height=${fixedCrop.height.toFixed(3)}`,
      ],
      tone: warnings.length === 0 ? "analysis-good" : "",
    },
    {
      title: "Quick Notes",
      items: warnings.length > 0
        ? warnings
        : ["Current alignment looks broadly reasonable for a first extraction pass."],
      tone: warnings.length > 0 ? "analysis-warn" : "analysis-good",
    },
  ];

  return cards;
}

function renderCropAnalysis() {
  const cards = buildCropAnalysis();
  els.cropAnalysisOutput.innerHTML = cards
    .map((card) => {
      const items = card.items
        .map((item) => `<li>${escapeHtml(item)}</li>`)
        .join("");
      return `
        <section class="analysis-card">
          <h3 class="${card.tone || ""}">${escapeHtml(card.title)}</h3>
          <ul>${items}</ul>
        </section>
      `;
    })
    .join("");
}

function updateControls() {
  const hasDocument = Boolean(state.documentId && state.previewImage);
  els.rotateLeft.disabled = !hasDocument || state.isBusy;
  els.rotateRight.disabled = !hasDocument || state.isBusy;
  els.resetAdjust.disabled = !hasDocument || state.isBusy;
  els.extractButton.disabled = !hasDocument || state.isBusy;
  els.uploadButton.disabled = state.isBusy;
  els.fileInput.disabled = state.isBusy;
  els.microRotate.disabled = !hasDocument || state.isBusy;
  els.zoomOut.disabled = !hasDocument || state.isBusy;
  els.zoomIn.disabled = !hasDocument || state.isBusy;
  els.zoomRange.disabled = !hasDocument || state.isBusy;
  els.offsetXRange.disabled = !hasDocument || state.isBusy;
  els.offsetYRange.disabled = !hasDocument || state.isBusy;
  els.rotationChip.textContent = `rotation: ${state.rotation}`;
  els.docIdChip.textContent = `document: ${state.documentId || "-"}`;
  els.microRotate.value = String(state.microRotation);
  els.zoomRange.value = String(state.zoom);
  els.offsetXRange.value = String(state.offsetX);
  els.offsetYRange.value = String(state.offsetY);
  updatePayloadView();
}

function updateDocumentSummary() {
  if (!state.upload) {
    els.docName.textContent = "No document";
    els.docMeta.textContent = "Upload a file to begin.";
    return;
  }

  els.docName.textContent = state.upload.filename;
  els.docMeta.textContent =
    `${state.upload.source_type} | ${state.upload.preview_width}x${state.upload.preview_height} | deduplicated=${state.upload.deduplicated}`;
}

function resetAdjustments() {
  state.microRotation = 0;
  state.zoom = 1;
  state.offsetX = 0;
  state.offsetY = 0;
  state.dragStart = null;
  state.dragCurrent = null;
  renderCanvas();
  renderCropAnalysis();
  updateControls();
}

function drawEmptyCanvas() {
  ctx.clearRect(0, 0, els.canvas.width, els.canvas.height);
  ctx.fillStyle = "#f6f0e5";
  ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);
  ctx.fillStyle = "#67757c";
  ctx.font = "600 18px Segoe UI";
  ctx.textAlign = "center";
  ctx.fillText("Upload a document to preview it here.", els.canvas.width / 2, els.canvas.height / 2);
}

function drawCropRect(rectPx) {
  ctx.save();
  ctx.fillStyle = "rgba(13, 107, 95, 0.16)";
  ctx.strokeStyle = "#0d6b5f";
  ctx.lineWidth = 2;
  ctx.fillRect(rectPx.x, rectPx.y, rectPx.width, rectPx.height);
  ctx.strokeRect(rectPx.x, rectPx.y, rectPx.width, rectPx.height);
  ctx.restore();
}

function drawOverlay(targetWidth, targetHeight) {
  const frameX = targetWidth * WORKING_FRAME_MARGIN_X;
  const frameY = targetHeight * WORKING_FRAME_MARGIN_Y;
  const frameWidth = targetWidth * (1 - (WORKING_FRAME_MARGIN_X * 2));
  const frameHeight = targetHeight * (1 - (WORKING_FRAME_MARGIN_Y * 2));
  const mrzFocusY = frameY + (frameHeight * (1 - MRZ_FOCUS_HEIGHT));
  const mrzFocusHeight = frameHeight * MRZ_FOCUS_HEIGHT;

  ctx.save();
  ctx.fillStyle = "rgba(40, 112, 197, 0.20)";
  ctx.beginPath();
  ctx.rect(0, 0, targetWidth, targetHeight);
  ctx.rect(frameX, frameY, frameWidth, frameHeight);
  ctx.fill("evenodd");
  ctx.strokeStyle = "rgba(34, 139, 34, 0.95)";
  ctx.lineWidth = 2;
  ctx.strokeRect(frameX, frameY, frameWidth, frameHeight);
  ctx.fillStyle = "rgba(40, 112, 197, 0.16)";
  ctx.fillRect(frameX, frameY, frameWidth, Math.max(0, mrzFocusY - frameY));
  ctx.strokeStyle = "rgba(20, 92, 170, 0.45)";
  ctx.setLineDash([8, 6]);
  ctx.strokeRect(frameX, mrzFocusY, frameWidth, mrzFocusHeight);
  ctx.restore();
}

function renderCanvasWithCanvas() {
  const source = getPaddedPreviewSource();
  const rotated = getRotatedImageSize();
  const targetWidth = Math.max(320, Math.round(els.viewerFrame.clientWidth - 2));
  const targetHeight = Math.max(240, Math.round(els.viewerFrame.clientHeight - 2));
  const scale = Math.min(targetWidth / rotated.width, targetHeight / rotated.height);

  els.canvas.width = targetWidth;
  els.canvas.height = targetHeight;
  state.canvasScale = scale;
  state.canvasBounds = { x: 0, y: 0, width: targetWidth, height: targetHeight };

  ctx.clearRect(0, 0, targetWidth, targetHeight);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.save();
  ctx.translate(targetWidth / 2, targetHeight / 2);
  ctx.scale(scale, scale);
  ctx.translate(state.offsetX * rotated.width, state.offsetY * rotated.height);
  ctx.rotate((state.microRotation * Math.PI) / 180);
  ctx.scale(state.zoom, state.zoom);

  if (state.rotation === 0) {
    ctx.drawImage(source, -source.width / 2, -source.height / 2);
  } else if (state.rotation === 90) {
    ctx.rotate(Math.PI / 2);
    ctx.drawImage(source, -source.width / 2, -source.height / 2);
  } else if (state.rotation === 180) {
    ctx.rotate(Math.PI);
    ctx.drawImage(source, -source.width / 2, -source.height / 2);
  } else if (state.rotation === 270) {
    ctx.rotate(-Math.PI / 2);
    ctx.drawImage(source, -source.width / 2, -source.height / 2);
  }
  ctx.restore();
  drawOverlay(targetWidth, targetHeight);
}

function renderCanvasWithOpenCv() {
  const cv = window.cv;
  const source = getPreviewMat();
  if (!source) {
    renderCanvasWithCanvas();
    return;
  }

  const rotatedBase = new cv.Mat();
  if (state.rotation === 90) {
    cv.rotate(source, rotatedBase, cv.ROTATE_90_CLOCKWISE);
  } else if (state.rotation === 180) {
    cv.rotate(source, rotatedBase, cv.ROTATE_180);
  } else if (state.rotation === 270) {
    cv.rotate(source, rotatedBase, cv.ROTATE_90_COUNTERCLOCKWISE);
  } else {
    source.copyTo(rotatedBase);
  }

  const targetWidth = Math.max(320, Math.round(els.viewerFrame.clientWidth - 2));
  const targetHeight = Math.max(240, Math.round(els.viewerFrame.clientHeight - 2));
  const h = rotatedBase.rows;
  const w = rotatedBase.cols;
  const padX = Math.round(w * TRANSFORM_PAD_RATIO);
  const padY = Math.round(h * TRANSFORM_PAD_RATIO);
  const padded = new cv.Mat();
  cv.copyMakeBorder(rotatedBase, padded, padY, padY, padX, padX, cv.BORDER_REPLICATE);

  const centerX = padded.cols / 2.0;
  const centerY = padded.rows / 2.0;
  const matrix = cv.getRotationMatrix2D(
    new cv.Point(centerX, centerY),
    -state.microRotation,
    state.zoom
  );
  matrix.doublePtr(0, 2)[0] += state.offsetX * w;
  matrix.doublePtr(1, 2)[0] += state.offsetY * h;

  const scale = Math.min(targetWidth / w, targetHeight / h);
  matrix.doublePtr(0, 0)[0] *= scale;
  matrix.doublePtr(0, 1)[0] *= scale;
  matrix.doublePtr(1, 0)[0] *= scale;
  matrix.doublePtr(1, 1)[0] *= scale;
  matrix.doublePtr(0, 2)[0] += (targetWidth / 2.0) - (centerX * scale);
  matrix.doublePtr(1, 2)[0] += (targetHeight / 2.0) - (centerY * scale);

  const rendered = new cv.Mat();
  cv.warpAffine(
    padded,
    rendered,
    matrix,
    new cv.Size(targetWidth, targetHeight),
    cv.INTER_LINEAR,
    cv.BORDER_CONSTANT,
    new cv.Scalar(255, 255, 255, 255)
  );

  els.canvas.width = targetWidth;
  els.canvas.height = targetHeight;
  state.canvasScale = scale;
  state.canvasBounds = { x: 0, y: 0, width: targetWidth, height: targetHeight };
  cv.imshow(els.canvas, rendered);
  drawOverlay(targetWidth, targetHeight);

  rendered.delete();
  matrix.delete();
  padded.delete();
  rotatedBase.delete();
}

function renderCanvas() {
  if (!state.previewImage) {
    els.viewerFrame.classList.add("empty");
    drawEmptyCanvas();
    return;
  }

  els.viewerFrame.classList.remove("empty");
  if (window.cv && cvState.ready) {
    renderCanvasWithOpenCv();
    return;
  }

  renderCanvasWithCanvas();
}

function getCanvasPointer(event) {
  const rect = els.canvas.getBoundingClientRect();
  const x = clamp(event.clientX - rect.left, 0, rect.width);
  const y = clamp(event.clientY - rect.top, 0, rect.height);
  return {
    x: rect.width === 0 ? 0 : x / rect.width,
    y: rect.height === 0 ? 0 : y / rect.height,
  };
}

async function loadPreviewImage(documentId) {
  const image = new Image();
  image.decoding = "async";
  image.src = `${getPreviewUrl(documentId)}?t=${Date.now()}`;
  await image.decode();
  return image;
}

async function handleUpload(event) {
  event.preventDefault();
  const file = els.fileInput.files[0];
  if (!file) {
    setStatus("Choose a file before uploading.", true);
    return;
  }

  state.isBusy = true;
  updateControls();
  setStatus(`Uploading ${file.name} ...`);

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/uploads", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Upload failed.");
    }

    const image = await loadPreviewImage(payload.document_id);
    state.upload = payload;
    state.documentId = payload.document_id;
    state.previewImage = image;
    clearPreviewMats();
    state.rotation = 0;
    state.microRotation = 0;
    state.zoom = 1;
    state.offsetX = 0;
    state.offsetY = 0;
    state.dragStart = null;
    state.dragCurrent = null;

    els.uploadJson.textContent = formatJson(payload);
    els.resultJson.textContent = "No extraction yet.";
    renderAnalysis(null);
    renderCropAnalysis();
    updateDocumentSummary();
    renderCanvas();
    setStatus(`Uploaded ${payload.filename}. Adjust rotation/crop, then run extraction.`);
  } catch (error) {
    setStatus(error.message || "Upload failed.", true);
  } finally {
    state.isBusy = false;
    updateControls();
  }
}

function rotate(delta) {
  if (!state.previewImage) {
    return;
  }
  state.rotation = (state.rotation + delta + 360) % 360;
  resetAdjustments();
  renderCanvas();
  renderCropAnalysis();
  updateControls();
  setStatus(`Rotation set to ${state.rotation} degrees.`);
}

async function handleExtraction() {
  if (!state.documentId) {
    return;
  }

  state.isBusy = true;
  updateControls();
  setStatus("Running extraction ...");

  try {
    const payload = buildExtractionPayload();
    const response = await fetch("/api/extractions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.detail || "Extraction failed.");
    }

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

function handlePointerDown(event) {
  if (!state.previewImage || state.isBusy) {
    return;
  }
  state.dragStart = getCanvasPointer(event);
  state.dragCurrent = state.dragStart;
  renderCanvas();
}

function handlePointerMove(event) {
  if (!state.dragStart) {
    return;
  }
  const next = getCanvasPointer(event);
  const dx = (next.x - state.dragCurrent.x) * DRAG_SENSITIVITY;
  const dy = (next.y - state.dragCurrent.y) * DRAG_SENSITIVITY;
  state.offsetX = clamp(state.offsetX + dx, -OFFSET_LIMIT, OFFSET_LIMIT);
  state.offsetY = clamp(state.offsetY + dy, -OFFSET_Y_LIMIT, OFFSET_Y_LIMIT);
  state.dragCurrent = next;
  renderCanvas();
  renderCropAnalysis();
  updateControls();
}

function handlePointerUp(event) {
  if (!state.dragStart) {
    return;
  }
  state.dragStart = null;
  state.dragCurrent = null;
  renderCanvas();
  renderCropAnalysis();
  updateControls();
  setStatus("Image adjustment updated.");
}

function handleResize() {
  renderCanvas();
}

function waitForOpenCv() {
  if (window.cv && typeof window.cv.Mat === "function") {
    markOpenCvReady();
    return;
  }

  if (window.cv && !cvState.hooked) {
    const previous = window.cv.onRuntimeInitialized;
    window.cv.onRuntimeInitialized = () => {
      if (typeof previous === "function") {
        previous();
      }
      markOpenCvReady();
    };
    cvState.hooked = true;
  }

  cvState.pollCount += 1;
  if (cvState.pollCount > 200) {
    cvState.failed = true;
    updateEngineChip();
    return;
  }

  window.setTimeout(waitForOpenCv, 150);
}

function init() {
  waitForOpenCv();
  updateEngineChip();
  drawEmptyCanvas();
  updateDocumentSummary();
  updateControls();
  renderAnalysis(null);
  renderCropAnalysis();
  els.uploadForm.addEventListener("submit", handleUpload);
  els.rotateLeft.addEventListener("click", () => rotate(270));
  els.rotateRight.addEventListener("click", () => rotate(90));
  els.resetAdjust.addEventListener("click", resetAdjustments);
  els.extractButton.addEventListener("click", handleExtraction);
  els.useFaceHint.addEventListener("change", updatePayloadView);
  els.microRotate.addEventListener("input", () => {
    state.microRotation = Number(els.microRotate.value);
    renderCanvas();
    renderCropAnalysis();
    updateControls();
  });
  els.zoomRange.addEventListener("input", () => {
    state.zoom = Number(els.zoomRange.value);
    renderCanvas();
    renderCropAnalysis();
    updateControls();
  });
  els.zoomOut.addEventListener("click", () => {
    state.zoom = Number(clamp(state.zoom - 0.03, ZOOM_MIN, ZOOM_MAX).toFixed(2));
    renderCanvas();
    renderCropAnalysis();
    updateControls();
  });
  els.zoomIn.addEventListener("click", () => {
    state.zoom = Number(clamp(state.zoom + 0.03, ZOOM_MIN, ZOOM_MAX).toFixed(2));
    renderCanvas();
    renderCropAnalysis();
    updateControls();
  });
  els.offsetXRange.addEventListener("input", () => {
    state.offsetX = Number(els.offsetXRange.value);
    renderCanvas();
    renderCropAnalysis();
    updateControls();
  });
  els.offsetYRange.addEventListener("input", () => {
    state.offsetY = Number(els.offsetYRange.value);
    renderCanvas();
    renderCropAnalysis();
    updateControls();
  });
  els.canvas.addEventListener("pointerdown", handlePointerDown);
  els.canvas.addEventListener("pointermove", handlePointerMove);
  els.canvas.addEventListener("pointerup", handlePointerUp);
  els.canvas.addEventListener("pointerleave", handlePointerUp);
  els.canvas.addEventListener("wheel", (event) => {
    if (!state.previewImage) {
      return;
    }
    event.preventDefault();
    const nextZoom = clamp(state.zoom + (event.deltaY < 0 ? 0.03 : -0.03), ZOOM_MIN, ZOOM_MAX);
    state.zoom = Number(nextZoom.toFixed(2));
    renderCanvas();
    renderCropAnalysis();
    updateControls();
  }, { passive: false });
  window.addEventListener("resize", handleResize);
}

init();
