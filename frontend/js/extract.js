import { els, setStatus, updateControls } from "./dom.js";
import { drawExportCanvas } from "./render.js";
import { formatJson, state } from "./state.js";
import { renderAnalysis } from "./analysis.js";

export function dataUrlToBlob(dataUrl) {
  const [meta, base64] = dataUrl.split(",");
  const mime = meta.match(/data:(.*?);base64/)?.[1] || "image/jpeg";
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return new Blob([bytes], { type: mime });
}

export function extractImage() {
  drawExportCanvas();
  return els.exportCanvas.toDataURL("image/jpeg", 0.95);
}

export function saveExportImage() {
  if (!state.previewImage) {
    return;
  }
  const imageDataUrl = extractImage();
  const link = document.createElement("a");
  const baseName = (state.upload?.filename || "document").replace(/\.[^.]+$/, "");
  link.href = imageDataUrl;
  link.download = `${baseName}_export_debug.jpg`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  setStatus(`Saved export debug image (${els.exportCanvas.width}x${els.exportCanvas.height}).`);
}

export async function handleExtraction() {
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

    setStatus("Running extraction ...");
    const response = await fetch("/api/extractions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        document_id: state.documentId,
        input_mode: "frontend",
        enable_correction: false,
        use_face_hint: false,
      }),
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
