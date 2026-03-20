import {
  QUALITY_LOW_CONTRAST_STDDEV,
  QUALITY_LOW_EDGE_STRENGTH,
  QUALITY_SAMPLE_MAX_WIDTH,
  qualityCanvas,
  qualityCtx,
  state,
} from "./state.js";
import { computeMrzGuideRect, getExportCanvasSize, getTransformedImageBounds, renderImage } from "./render.js";

export function computeImageQuality() {
  if (!state.previewImage) {
    return null;
  }

  const sampleWidth = Math.max(96, Math.min(QUALITY_SAMPLE_MAX_WIDTH, Math.round(state.viewSize.width / 4)));
  const sampleHeight = Math.max(72, Math.round(sampleWidth * (state.viewSize.height / state.viewSize.width)));
  const scaleRatio = sampleWidth / state.viewSize.width;

  qualityCanvas.width = sampleWidth;
  qualityCanvas.height = sampleHeight;
  renderImage(qualityCtx, qualityCanvas, state.viewBaseScale * scaleRatio, "#ffffff");

  const imageData = qualityCtx.getImageData(0, 0, sampleWidth, sampleHeight).data;
  let luminanceSum = 0;
  let luminanceSumSq = 0;
  let luminanceCount = 0;
  let edgeSum = 0;
  let edgeCount = 0;

  const step = 2;
  for (let y = 0; y < sampleHeight; y += step) {
    for (let x = 0; x < sampleWidth; x += step) {
      const index = ((y * sampleWidth) + x) * 4;
      const r = imageData[index];
      const g = imageData[index + 1];
      const b = imageData[index + 2];
      const lum = (0.299 * r) + (0.587 * g) + (0.114 * b);
      luminanceSum += lum;
      luminanceSumSq += lum * lum;
      luminanceCount += 1;

      if (x + step < sampleWidth) {
        const rightIndex = ((y * sampleWidth) + (x + step)) * 4;
        const rightLum = (0.299 * imageData[rightIndex]) + (0.587 * imageData[rightIndex + 1]) + (0.114 * imageData[rightIndex + 2]);
        edgeSum += Math.abs(lum - rightLum);
        edgeCount += 1;
      }
      if (y + step < sampleHeight) {
        const downIndex = ((((y + step) * sampleWidth) + x)) * 4;
        const downLum = (0.299 * imageData[downIndex]) + (0.587 * imageData[downIndex + 1]) + (0.114 * imageData[downIndex + 2]);
        edgeSum += Math.abs(lum - downLum);
        edgeCount += 1;
      }
    }
  }

  const mean = luminanceSum / luminanceCount;
  const variance = Math.max(0, (luminanceSumSq / luminanceCount) - (mean * mean));
  const stddev = Math.sqrt(variance);
  const edgeStrength = edgeCount > 0 ? (edgeSum / edgeCount) : 0;

  return {
    stddev,
    edgeStrength,
    exportSize: getExportCanvasSize(),
  };
}

export function buildQualityCard() {
  const metrics = computeImageQuality();
  if (!metrics) {
    return {
      title: "Input Quality",
      items: ["Load a document to inspect image quality."],
      tone: "",
    };
  }

  const warnings = [];
  if (metrics.stddev < QUALITY_LOW_CONTRAST_STDDEV) {
    warnings.push(`Low contrast: ${metrics.stddev.toFixed(1)} luminance stddev.`);
  }
  if (metrics.edgeStrength < QUALITY_LOW_EDGE_STRENGTH) {
    warnings.push(`Likely blur: ${metrics.edgeStrength.toFixed(1)} edge strength.`);
  }
  if (metrics.exportSize.width < 600 || metrics.exportSize.height < 400) {
    warnings.push(`Export is ${metrics.exportSize.width}x${metrics.exportSize.height}, which may be rejected by the backend minimum size check.`);
  }

  const guide = computeMrzGuideRect();
  const bounds = getTransformedImageBounds();
  if (!bounds) {
    warnings.push("No page bounds detected yet.");
  } else {
    const overlapsGuide =
      bounds.left < guide.x + guide.width &&
      bounds.right > guide.x &&
      bounds.top < guide.y + guide.height &&
      bounds.bottom > guide.y;

    if (!overlapsGuide) {
      warnings.push("The MRZ band is not overlapping the guide box yet.");
    } else if (bounds.bottom < guide.y + (guide.height * 0.6)) {
      warnings.push("Move the page a little lower so the MRZ band sits closer to the guide.");
    }
  }

  return {
    title: "Input Quality",
    items: warnings.length > 0
      ? warnings
      : ["Blur, contrast, and placement look acceptable for a frontend-prepared image."],
    tone: warnings.length > 0 ? "analysis-warn" : "analysis-good",
  };
}
