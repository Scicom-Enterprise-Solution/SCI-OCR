import {
  FACE_ANALYSIS_INTERVAL_MS,
  GUIDANCE_ANALYSIS_INTERVAL_MS,
  GUIDANCE_LOCK_MISS_LIMIT,
  guidanceCanvas,
  guidanceCtx,
  state,
} from "./state.js";
import { updateControls } from "./dom.js";
import { computeMrzGuideRect, getTransformedImageBounds, renderGuidanceSource } from "./render.js";

export function getGuideStatus() {
  if (!state.previewImage) {
    return "No document loaded.";
  }

  if (state.guidance.mrzRect && isGuideCoveredByPage()) {
    return state.guidance.faceRect ? "MRZ aligned and face detected" : "MRZ aligned";
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

function isGuideCoveredByPage() {
  const bounds = getTransformedImageBounds();
  if (!bounds) {
    return false;
  }
  const guide = computeMrzGuideRect();
  const leftGapRatio = Math.max(0, (guide.x - bounds.left) / guide.width);
  const rightGapRatio = Math.max(0, (bounds.right - (guide.x + guide.width)) < 0 ? ((guide.x + guide.width) - bounds.right) / guide.width : 0);
  const totalGapRatio = leftGapRatio + rightGapRatio;

  return leftGapRatio <= 0.12 && rightGapRatio <= 0.18 && totalGapRatio <= 0.2;
}

export function ensureCvReady() {
  if (typeof window.cv === "undefined") {
    return false;
  }
  if (window.cv && typeof window.cv.Mat === "function") {
    state.guidance.cvStatus = "ready";
    return true;
  }
  if (window.cv && typeof window.cv.onRuntimeInitialized !== "undefined") {
    window.cv.onRuntimeInitialized = () => {
      state.guidance.cvStatus = "ready";
      updateControls();
    };
  }
  return false;
}

function detectProjectionCandidate(cv, thresh, analysisRect, guideRect) {
  const guideLocalX = guideRect.x - analysisRect.x;
  const guideLocalY = guideRect.y - analysisRect.y;
  const safeWidth = Math.min(guideRect.width, thresh.cols - guideLocalX);
  const safeHeight = Math.min(guideRect.height, thresh.rows - guideLocalY);
  if (safeWidth < 20 || safeHeight < 12) {
    return null;
  }

  const roiRect = new cv.Rect(guideLocalX, guideLocalY, safeWidth, safeHeight);
  let guideThresh = null;
  try {
    guideThresh = thresh.roi(roiRect);
    const rowThreshold = Math.max(8, Math.round(safeWidth * 0.05));
    const colThreshold = Math.max(3, Math.round(safeHeight * 0.08));

    let top = -1;
    let bottom = -1;
    for (let y = 0; y < safeHeight; y += 1) {
      const rowRect = new cv.Rect(0, y, safeWidth, 1);
      let row = null;
      try {
        row = guideThresh.roi(rowRect);
        const count = cv.countNonZero(row);
        if (count >= rowThreshold) {
          if (top === -1) {
            top = y;
          }
          bottom = y;
        }
      } finally {
        if (row) {
          row.delete();
        }
      }
    }

    let left = -1;
    let right = -1;
    for (let x = 0; x < safeWidth; x += 1) {
      const colRect = new cv.Rect(x, 0, 1, safeHeight);
      let col = null;
      try {
        col = guideThresh.roi(colRect);
        const count = cv.countNonZero(col);
        if (count >= colThreshold) {
          if (left === -1) {
            left = x;
          }
          right = x;
        }
      } finally {
        if (col) {
          col.delete();
        }
      }
    }

    if (top === -1 || bottom === -1 || left === -1 || right === -1) {
      return null;
    }

    const width = right - left + 1;
    const height = bottom - top + 1;
    const widthRatio = width / guideRect.width;
    const heightRatio = height / guideRect.height;
    const leftGapRatio = left / guideRect.width;
    const rightGapRatio = (safeWidth - (right + 1)) / guideRect.width;
    const totalGapRatio = leftGapRatio + rightGapRatio;

    if (widthRatio < 0.78 || widthRatio > 1.02) {
      return null;
    }
    if (heightRatio < 0.22 || heightRatio > 0.72) {
      return null;
    }
    if (leftGapRatio > 0.12 || rightGapRatio > 0.18 || totalGapRatio > 0.22) {
      return null;
    }

    return {
      x: guideRect.x + left,
      y: guideRect.y + top,
      width,
      height,
      score: (widthRatio * 3.8) - (totalGapRatio * 1.6) - (Math.abs(heightRatio - 0.48) * 0.8) + 0.5,
      source: "projection",
    };
  } finally {
    if (guideThresh) {
      guideThresh.delete();
    }
  }
}

export function detectMrzGuideRect() {
  if (!ensureCvReady() || !state.previewImage) {
    return null;
  }

  renderGuidanceSource(guidanceCanvas, guidanceCtx);
  const guide = computeMrzGuideRect();
  const cv = window.cv;
  let src = null;
  let roi = null;
  let gray = null;
  let blackhat = null;
  let gradX = null;
  let absGradX = null;
  let blurred = null;
  let thresh = null;
  let kernel = null;
  let closeKernel = null;
  let contours = null;
  let hierarchy = null;
  let erodeKernel = null;
  let dilateKernel = null;
  const EDGE_BAND_RATIO = 0.12;
  const EDGE_DENSITY_MIN = 0.014;

  try {
    src = cv.imread(guidanceCanvas);
    const guideRect = new cv.Rect(
      Math.max(0, Math.round(guide.x)),
      Math.max(0, Math.round(guide.y)),
      Math.min(src.cols - Math.round(guide.x), Math.round(guide.width)),
      Math.min(src.rows - Math.round(guide.y), Math.round(guide.height)),
    );

    const padLeft = Math.round(guideRect.width * 0.06);
    const padRight = Math.round(guideRect.width * 0.14);
    const padTop = Math.round(guideRect.height * 0.18);
    const padBottom = Math.round(guideRect.height * 0.14);
    const analysisRect = new cv.Rect(
      Math.max(0, guideRect.x - padLeft),
      Math.max(0, guideRect.y - padTop),
      Math.min(src.cols - Math.max(0, guideRect.x - padLeft), guideRect.width + padLeft + padRight),
      Math.min(src.rows - Math.max(0, guideRect.y - padTop), guideRect.height + padTop + padBottom),
    );

    if (guideRect.width < 20 || guideRect.height < 10 || analysisRect.width < 20 || analysisRect.height < 10) {
      return null;
    }

    roi = src.roi(analysisRect);
    gray = new cv.Mat();
    cv.cvtColor(roi, gray, cv.COLOR_RGBA2GRAY);

    kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(Math.max(15, Math.round(analysisRect.width * 0.12)), 5));
    blackhat = new cv.Mat();
    cv.morphologyEx(gray, blackhat, cv.MORPH_BLACKHAT, kernel);

    gradX = new cv.Mat();
    cv.Sobel(blackhat, gradX, cv.CV_32F, 1, 0, 3, 1, 0, cv.BORDER_DEFAULT);
    absGradX = new cv.Mat();
    cv.convertScaleAbs(gradX, absGradX);

    blurred = new cv.Mat();
    cv.GaussianBlur(absGradX, blurred, new cv.Size(5, 5), 0, 0, cv.BORDER_DEFAULT);

    thresh = new cv.Mat();
    cv.threshold(blurred, thresh, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);

    closeKernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(Math.max(21, Math.round(analysisRect.width * 0.18)), 7));
    erodeKernel = new cv.Mat();
    dilateKernel = new cv.Mat();
    cv.morphologyEx(thresh, thresh, cv.MORPH_CLOSE, closeKernel);
    cv.erode(thresh, thresh, erodeKernel, new cv.Point(-1, -1), 1);
    cv.dilate(thresh, thresh, dilateKernel, new cv.Point(-1, -1), 2);

    contours = new cv.MatVector();
    hierarchy = new cv.Mat();
    cv.findContours(thresh, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    let best = null;
    for (let i = 0; i < contours.size(); i += 1) {
      const contour = contours.get(i);
      const rect = cv.boundingRect(contour);
      contour.delete();

      const candidateX = analysisRect.x + rect.x;
      const candidateY = analysisRect.y + rect.y;
      const widthRatio = rect.width / guideRect.width;
      const heightRatio = rect.height / guideRect.height;
      const aspect = rect.width / Math.max(1, rect.height);
      if (widthRatio < 0.72 || widthRatio > 1.02) {
        continue;
      }
      if (heightRatio < 0.12 || heightRatio > 0.62) {
        continue;
      }
      if (aspect < 4.5) {
        continue;
      }

      const overlapsGuideHorizontally =
        candidateX < guideRect.x + guideRect.width &&
        candidateX + rect.width > guideRect.x;
      const overlapsGuideVertically =
        candidateY < guideRect.y + guideRect.height &&
        candidateY + rect.height > guideRect.y;
      if (!overlapsGuideHorizontally || !overlapsGuideVertically) {
        continue;
      }

      const leftGapRatio = Math.max(0, (candidateX - guideRect.x) / guideRect.width);
      const rightGapRatio = Math.max(0, ((guideRect.x + guideRect.width) - (candidateX + rect.width)) / guideRect.width);
      const totalGapRatio = leftGapRatio + rightGapRatio;
      const maxGapRatio = Math.max(leftGapRatio, rightGapRatio);
      if (leftGapRatio > 0.16 || rightGapRatio > 0.2 || totalGapRatio > 0.22) {
        continue;
      }

      const guideLocalX = guideRect.x - analysisRect.x;
      const guideLocalY = guideRect.y - analysisRect.y;
      const bandWidth = Math.max(10, Math.round(guideRect.width * EDGE_BAND_RATIO));
      const rightBandWidth = Math.max(bandWidth, Math.round(guideRect.width * 0.16));
      const bandTop = Math.max(rect.y, guideLocalY);
      const bandBottom = Math.min(rect.y + rect.height, guideLocalY + guideRect.height);
      const bandHeight = bandBottom - bandTop;
      if (bandHeight < 6) {
        continue;
      }

      const leftBandRect = new cv.Rect(
        guideLocalX,
        bandTop,
        Math.min(bandWidth, thresh.cols - guideLocalX),
        bandHeight,
      );
      const rightBandX = Math.max(guideLocalX, (guideLocalX + guideRect.width) - bandWidth);
      const rightBandRect = new cv.Rect(
        rightBandX,
        bandTop,
        Math.min(rightBandWidth, thresh.cols - rightBandX),
        bandHeight,
      );

      let leftDensity = 0;
      let rightDensity = 0;
      let leftBand = null;
      let rightBand = null;
      try {
        leftBand = thresh.roi(leftBandRect);
        rightBand = thresh.roi(rightBandRect);
        leftDensity = cv.countNonZero(leftBand) / (leftBandRect.width * leftBandRect.height);
        rightDensity = cv.countNonZero(rightBand) / (rightBandRect.width * rightBandRect.height);
      } finally {
        if (leftBand) {
          leftBand.delete();
        }
        if (rightBand) {
          rightBand.delete();
        }
      }

      if (leftDensity < EDGE_DENSITY_MIN || rightDensity < EDGE_DENSITY_MIN) {
        continue;
      }

      const centerYRatio = ((candidateY + (rect.height / 2)) - guideRect.y) / guideRect.height;
      const centerXRatio = ((candidateX + (rect.width / 2)) - guideRect.x) / guideRect.width;
      const leftAnchorRatio = Math.max(0, Math.min(1, (candidateX - guideRect.x) / Math.max(1, guideRect.width * 0.12)));
      const rightAnchorRatio = Math.max(0, Math.min(1, ((guideRect.x + guideRect.width) - (candidateX + rect.width)) / Math.max(1, guideRect.width * 0.16)));
      const score =
        (widthRatio * 3.4) +
        (Math.min(aspect, 14) * 0.08) -
        Math.abs(centerYRatio - 0.55) -
        Math.abs(centerXRatio - 0.5) * 0.35 -
        (totalGapRatio * 1.5) -
        (maxGapRatio * 0.75) -
        (Math.min(leftDensity, rightDensity) * 0.6) -
        (leftAnchorRatio * 0.35) -
        (rightAnchorRatio * 0.2) -
        (heightRatio * 0.2);
      if (!best || score > best.score) {
        best = {
          x: candidateX,
          y: candidateY,
          width: rect.width,
          height: rect.height,
          score,
          source: "contour",
        };
      }
    }

    const projectionCandidate = detectProjectionCandidate(cv, thresh, analysisRect, guideRect);
    if (projectionCandidate && (!best || projectionCandidate.score >= best.score - 0.2)) {
      return projectionCandidate;
    }

    return best;
  } catch (error) {
    state.guidance.cvStatus = "error";
    return null;
  } finally {
    [
      src,
      roi,
      gray,
      blackhat,
      gradX,
      absGradX,
      blurred,
      thresh,
      kernel,
      closeKernel,
      contours,
      hierarchy,
      erodeKernel,
      dilateKernel,
    ]
      .filter(Boolean)
      .forEach((mat) => mat.delete());
  }
}

export async function detectFaceGuidance(timestampMs) {
  if (!state.previewImage || typeof window.FaceDetector === "undefined") {
    return;
  }
  if (state.guidance.facePending || (timestampMs - state.guidance.lastFaceMs) < FACE_ANALYSIS_INTERVAL_MS) {
    return;
  }

  state.guidance.facePending = true;
  state.guidance.lastFaceMs = timestampMs;
  renderGuidanceSource(guidanceCanvas, guidanceCtx);

  try {
    const detector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
    const faces = await detector.detect(guidanceCanvas);
    if (faces.length > 0) {
      const face = faces
        .map((entry) => ({
          x: entry.boundingBox.x,
          y: entry.boundingBox.y,
          width: entry.boundingBox.width,
          height: entry.boundingBox.height,
          area: entry.boundingBox.width * entry.boundingBox.height,
        }))
        .sort((a, b) => b.area - a.area)[0];
      state.guidance.faceRect = face;
      state.guidance.faceConfidence = 1;
    } else {
      state.guidance.faceRect = null;
      state.guidance.faceConfidence = 0;
    }
  } catch (error) {
    state.guidance.faceRect = null;
    state.guidance.faceConfidence = 0;
  } finally {
    state.guidance.facePending = false;
  }
}

export function runGuidanceAnalysis(timestampMs) {
  if (!state.previewImage) {
    state.guidance.mrzRect = null;
    state.guidance.faceRect = null;
    return;
  }

  if (!isGuideCoveredByPage()) {
    state.guidance.mrzRect = null;
    state.guidance.mrzScore = 0;
    state.guidance.mrzMisses = 0;
    void detectFaceGuidance(timestampMs);
    return;
  }

  if (ensureCvReady()) {
    if ((timestampMs - state.guidance.lastAnalysisMs) >= GUIDANCE_ANALYSIS_INTERVAL_MS) {
      state.guidance.lastAnalysisMs = timestampMs;
      const mrz = detectMrzGuideRect();
      if (mrz) {
        state.guidance.mrzRect = mrz;
        state.guidance.mrzScore = mrz.score;
        state.guidance.mrzMisses = 0;
      } else if (state.guidance.mrzRect) {
        state.guidance.mrzMisses += 1;
        if (state.guidance.mrzMisses >= GUIDANCE_LOCK_MISS_LIMIT) {
          state.guidance.mrzRect = null;
          state.guidance.mrzScore = 0;
          state.guidance.mrzMisses = 0;
        }
      } else {
        state.guidance.mrzRect = null;
        state.guidance.mrzScore = 0;
      }
    }
  }

  void detectFaceGuidance(timestampMs);
}

export function buildGuidanceCard() {
  const cvState = state.guidance.cvStatus === "ready"
    ? "OpenCV.js ready"
    : state.guidance.cvStatus === "error"
      ? "OpenCV.js unavailable"
      : "OpenCV.js loading";

  const items = [
    `MRZ guidance engine: ${cvState}`,
    state.guidance.mrzRect
      ? `MRZ candidate locked with score ${state.guidance.mrzScore.toFixed(2)}`
      : "MRZ candidate not locked yet",
  ];

  if (typeof window.FaceDetector === "undefined") {
    items.push("Face guidance API unavailable in this browser");
  } else if (state.guidance.faceRect) {
    items.push("Face detected for placement reference");
  } else {
    items.push("Face not detected yet");
  }

  return {
    title: "Live Guidance",
    items,
    tone: state.guidance.mrzRect ? "analysis-good" : "",
  };
}
