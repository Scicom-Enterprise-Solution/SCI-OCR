import { els } from "./dom.js";
import { buildGuidanceCard, getGuideStatus } from "./guidance.js";
import { buildQualityCard } from "./quality.js";
import { computeMrzGuideRect, getExportCanvasSize, getTransformedImageBounds } from "./render.js";
import { escapeHtml, state } from "./state.js";

export function buildAnalysis(result) {
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

export function renderAnalysis(result) {
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

export function buildAdjustmentCards() {
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
        `Rotation: ${state.rotation.toFixed(1)} degrees`,
        `Fine angle: ${state.fineRotation.toFixed(1)} degrees`,
        `Scale: ${state.scale.toFixed(2)}x`,
        `Offset: x=${state.offsetX.toFixed(3)}, y=${state.offsetY.toFixed(3)}`,
        `View canvas: ${state.viewSize.width}x${state.viewSize.height}`,
        `Export canvas: ${getExportCanvasSize().width}x${getExportCanvasSize().height}`,
      ],
      tone: "analysis-good",
    },
    buildQualityCard(),
    buildGuidanceCard(),
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

export function renderCropAnalysis() {
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
