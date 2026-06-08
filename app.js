const explorerBase = "https://huggingface.co/spaces/EEEAILab/ICAExplorer";

const bankParagraph =
  "Maya stopped at the bank before the trip, waiting in line to deposit a check and withdraw enough cash for the weekend. " +
  "The teller asked about her travel plans, and Maya said she was driving north to visit the old village by the river. " +
  "By sunset she was sitting on the grassy bank with her shoes beside her, watching the water curl around stones and reeds. " +
  "It amused her that the same word had followed her all day: first a bank with counters, accounts, and vaults, then a bank of earth holding the river in place.\n";

const libraryText =
  "She arrived at the library to study for her exam. Maya stopped at the bank before the trip, waiting in line to deposit a check and withdraw enough cash for the weekend.";

const analogyText =
  "king queen man woman father mother actor actress prince princess doctor nurse";

const cases = [
  {
    title: "Same word, different mixtures",
    shortTitle: "Bank polysemy",
    model: "gpt2",
    modelLabel: "GPT-2 Small",
    layer: "layer_06",
    topK: 5,
    cardWidth: 140,
    opacityCutoff: 0.5,
    components: "67:-",
    text: bankParagraph,
    highlights: ["bank", "deposit", "cash", "river", "bank", "vaults", "earth"],
    summary:
      "Follow the repeated word bank as it moves from finance to river geography. The selected component traces part of the broader context.",
  },
  {
    title: "A context-sensitive trace across a sentence",
    shortTitle: "Span trace",
    model: "gpt2",
    modelLabel: "GPT-2 Small",
    layer: "layer_06",
    topK: 5,
    cardWidth: 140,
    opacityCutoff: 0.5,
    components: "67:+,273:+",
    text: libraryText,
    highlights: ["arrived", "library", "study", "bank", "deposit", "cash"],
    summary:
      "Some ICA directions look less like isolated detectors and more like contextual factors that rise and fall across related tokens.",
  },
  {
    title: "Middle-layer context in Gemma 2 2B",
    shortTitle: "Gemma case",
    model: "gemma2_2b",
    modelLabel: "Gemma 2 2B",
    layer: "layer_12",
    topK: 5,
    cardWidth: 140,
    opacityCutoff: 0.5,
    components: "",
    text: bankParagraph,
    highlights: ["Maya", "bank", "teller", "river", "grassy", "bank"],
    summary:
      "Open the same paragraph in a different model family and compare which components surface in the middle residual stream.",
  },
  {
    title: "Embedding-level lexical structure",
    shortTitle: "Analogy tokens",
    model: "qwen3_5_2b_base",
    modelLabel: "Qwen 3.5 2B Base",
    layer: "embedding",
    topK: 5,
    cardWidth: 140,
    opacityCutoff: 0.5,
    components: "",
    text: analogyText,
    highlights: ["king", "queen", "father", "mother", "actor", "actress"],
    summary:
      "ICA can also be used on embeddings, where familiar word sets reveal overlapping lexical, gender-associated, family, and morphology directions.",
  },
];

function explorerUrl(item) {
  const params = new URLSearchParams({
    model: item.model,
    layer: item.layer,
    text: item.text,
    top_k: String(item.topK),
    card_width: String(item.cardWidth),
    opacity_cutoff: String(item.opacityCutoff),
  });

  if (item.components) {
    params.set("components", item.components);
  }

  return `${explorerBase}?${params.toString()}`;
}

function renderTokens(item) {
  const strip = document.querySelector("#token-strip");
  strip.innerHTML = "";

  item.highlights.forEach((token, index) => {
    const span = document.createElement("span");
    span.className = "token";
    span.textContent = token;
    if (index === 1 || index === 4 || token === "deposit" || token === "river") {
      span.classList.add("hot");
    }
    strip.appendChild(span);
  });
}

function selectCase(index) {
  const item = cases[index];
  const url = explorerUrl(item);

  document.querySelectorAll(".case-card").forEach((card, cardIndex) => {
    card.classList.toggle("active", cardIndex === index);
  });

  document.querySelector("#preview-model").textContent = `${item.modelLabel}, ${item.layer.replace("_", " ")}`;
  document.querySelector("#preview-title").textContent = item.title;
  document.querySelector("#preview-copy").textContent = item.summary;
  document.querySelector("#preview-link").href = url;
  renderTokens(item);
}

function renderCases() {
  const list = document.querySelector("#case-list");
  list.innerHTML = "";

  cases.forEach((item, index) => {
    const button = document.createElement("button");
    button.className = "case-card";
    button.type = "button";
    button.innerHTML = `
      <div class="case-meta">
        <span class="pill">${item.modelLabel}</span>
        <span class="pill alt">${item.layer.replace("_", " ")}</span>
        <span class="pill blue">top ${item.topK}</span>
      </div>
      <h3>${item.shortTitle}</h3>
      <p>${item.summary}</p>
    `;
    button.addEventListener("click", () => selectCase(index));
    list.appendChild(button);
  });
}

function initLinks() {
  const firstCase = explorerUrl(cases[0]);
  document.querySelector("#hero-case-link").href = firstCase;
  document.querySelector("#bank-inline-link").href = firstCase;
}

renderCases();
initLinks();
selectCase(0);
