const explorerBase = "https://huggingface.co/spaces/EEEAILab/ICAExplorer";
const saeExplorerBase = "https://eeeailab-icaexplorer.hf.space/sae-explorer";

const bankParagraph =
  "Maya stopped at the bank before the trip, waiting in line to deposit a check and withdraw enough cash for the weekend. " +
  "The teller asked about her travel plans, and Maya said she was driving north to visit the old village by the river. " +
  "By sunset she was sitting on the grassy bank with her shoes beside her, watching the water curl around stones and reeds. " +
  "It amused her that the same word had followed her all day: first a bank with counters, accounts, and vaults, then a bank of earth holding the river in place.\n";

const bankCase = {
  model: "gpt2",
  layer: "layer_06",
  topK: 5,
  cardWidth: 140,
  opacityCutoff: 0.5,
  components: "67:-",
  text: bankParagraph,
};

const geopoliticalText =
  "The US president threatened to tear up the nuclear deal between Tehran and major powers. " +
  "The university president threatened to cancel the agreement between Boston and nearby colleges. " +
  "Officials debated whether military action against North Korea would cross a red line. " +
  "Officials debated whether highway spending in North Dakota would cross the budget line.";

const lowLevelCodeText =
  "#define MDIO_CMD 0x20\n" +
  "uint32_t flags = 0;\n" +
  "struct Msghdr { uint8 Len; }\n" +
  "class Message { length: number; }\n" +
  "The PLL setting is one.";

const chineseLiteraryText =
  "山雨将至，旧城灯火微微，世间忧喜都在风里。" +
  "报告指出，居民担忧交通拥堵问题。" +
  "月色虽寒，江边的人影仍慢慢向前。" +
  "设备虽小，但可以正常工作。" +
  "阴云压着远山，旅人听见门外的马蹄声。";

const studyProtocolText =
  "Informed consent was obtained, and the institutional review board approved the study. " +
  "The institutional policy changed after the budget meeting. " +
  "The study protocol was approved by the ethics committee. " +
  "The travel plan was approved by the finance office. " +
  "All participants provided written informed consent before enrollment.";

const cases = [
  {
    title: "GPT-2: geopolitical conflict contexts",
    shortTitle: "Geopolitical conflict",
    model: "gpt2",
    modelLabel: "GPT-2 Small",
    layer: "layer_06",
    topK: 5,
    cardWidth: 140,
    opacityCutoff: 0.5,
    components: "290:+",
    text: geopoliticalText,
    highlights: ["Tehran", "Boston", "North", "Korea", "Dakota", "military", "nuclear"],
    summary:
      "A random-audit component labeled geopolitical conflicts: compare Iran/North Korea contexts with superficially similar city or state mentions.",
  },
  {
    title: "Gemma 2 2B: low-level programming tokens",
    shortTitle: "Low-level code",
    model: "gemma2_2b",
    modelLabel: "Gemma 2 2B",
    layer: "layer_16",
    topK: 5,
    cardWidth: 140,
    opacityCutoff: 0.5,
    components: "206:+",
    text: lowLevelCodeText,
    highlights: ["#define", "0x20", "uint32_t", "struct", "uint8", "class", "PLL"],
    summary:
      "A code-oriented component that prefers C/ASM-like tokens such as hex literals, uint types, structs, and register-style names.",
  },
  {
    title: "Qwen: Chinese atmospheric literary text",
    shortTitle: "Chinese literary style",
    model: "qwen3_5_2b_base",
    modelLabel: "Qwen 3.5 2B Base",
    layer: "layer_19",
    topK: 5,
    cardWidth: 140,
    opacityCutoff: 0.5,
    components: "808:-",
    text: chineseLiteraryText,
    highlights: ["忧", "担忧", "虽", "设备", "阴云", "旅人", "马蹄声"],
    summary:
      "A negative-side component from the audit: literary, atmospheric Chinese characters contrast with more literal news or instruction uses.",
  },
  {
    title: "Qwen: clinical study protocol language",
    shortTitle: "Study protocol",
    model: "qwen3_5_2b_base",
    modelLabel: "Qwen 3.5 2B Base",
    layer: "layer_18",
    topK: 5,
    cardWidth: 140,
    opacityCutoff: 0.5,
    components: "1794:+",
    text: studyProtocolText,
    highlights: ["consent", "institutional", "review", "protocol", "approved", "ethics", "participants"],
    summary:
      "A clinical-study component that separates IRB approval, study protocol, and informed-consent language from matched ordinary uses.",
  },
];


const overlapExamples = {
  gpt2: {
    model: "gpt2",
    layer: "layer_10",
    topK: 5,
    cardWidth: 140,
    opacityCutoff: 0.5,
    components: "142:-",
    saeFeature: 29658,
    text: `string "You're pretty good.$"

Route110_Text_16EA2A:: @ 816EA2`,
  },
  gemma: {
    model: "gemma2_2b",
    layer: "layer_24",
    topK: 5,
    cardWidth: 140,
    opacityCutoff: 0.5,
    components: "510:+",
    saeFeature: 9501,
    text: "US Open Final. Many of her fans and supporters have felt that the referee, Carlos Ramos have wronged Serena Williams and took from",
  },
  qwen: {
    model: "qwen3_5_2b_base",
    layer: "layer_22",
    topK: 5,
    cardWidth: 140,
    opacityCutoff: 0.5,
    components: "511:-",
    saeFeature: 8570,
    text: "'s an onScroll event that you can use as a trigger, here's a post on how to determine whether they scrolled up",
  },
};

const validationCases = [
  {
    title: "After",
    componentLabel: "L0/C192",
    model: "gpt2",
    modelLabel: "GPT-2 Small",
    layer: "layer_00",
    component: 192,
    side: "negative",
    components: "192:-",
    confidence: "high",
    type: "Word",
    erf: "1.0",
    kurtosis: "298",
    rows: [
      { expected: "activate", prompt: "I went outside to play after I finished my homework.", target: "after", score: "-22.447", rank: "1" },
      { expected: "activate", prompt: "The puppy ran after the ball.", target: "after", score: "-22.700", rank: "1" },
      { expected: "not activate", prompt: "I went outside to play when I finished my homework.", target: "when", score: "-1.017", rank: "18" },
      { expected: "not activate", prompt: "The puppy ran through the ball.", target: "through", score: "+1.503", rank: "21" },
    ],
  },
  {
    title: "Scientific research / citation",
    componentLabel: "L7/C17",
    model: "gpt2",
    modelLabel: "GPT-2 Small",
    layer: "layer_07",
    component: 17,
    side: "negative",
    components: "17:-",
    confidence: "high",
    type: "Sentence",
    erf: "2.0",
    kurtosis: "33",
    rows: [
      { expected: "activate", prompt: "Smith, J., & Lee, K. (2021).", target: ").", score: "-9.480", rank: "1" },
      { expected: "activate", prompt: "Vaswani, A., et al. (2017). Attention is all you need.", target: ".", score: "-16.003", rank: "1" },
      { expected: "not activate", prompt: "Smith and Lee (2021) is a fictional patent citation used here for illustration purposes only.", target: ".", score: "-2.051", rank: "14" },
      { expected: "not activate", prompt: "In 2017, Ashish Vaswani led the team that published Attention Is All You Need, introducing the Transformer model.", target: ".", score: "+0.114", rank: "639" },
    ],
  },
  {
    title: "Gaming language",
    componentLabel: "L10/C368",
    model: "gpt2",
    modelLabel: "GPT-2 Small",
    layer: "layer_10",
    component: 368,
    side: "positive",
    components: "368:+",
    confidence: "high",
    type: "Global",
    erf: "3.2",
    kurtosis: "31",
    rows: [
      { expected: "activate", prompt: "Don't fight before dragon unless our jungler has Smite up; save your ult for their engage.", target: "their", score: "+21.492", rank: "1" },
      { expected: "activate", prompt: "If the tank does not kite during the second phase, the healers get overwhelmed by the damage.", target: "by", score: "+16.429", rank: "1" },
      { expected: "not activate", prompt: "Don't leave before dinner unless everyone has arrived; save your announcement for their arrival.", target: "their", score: "+0.158", rank: "634" },
      { expected: "not activate", prompt: "If the manager does not delegate during the busiest part of the project, the team gets overwhelmed by the workload.", target: "by", score: "-0.328", rank: "471" },
    ],
  },
  {
    title: "Prior section-header repetition",
    componentLabel: "L8/C738",
    model: "gpt2",
    modelLabel: "GPT-2 Small",
    layer: "layer_08",
    component: 738,
    side: "negative",
    components: "738:-",
    confidence: "medium",
    type: "Long-range",
    erf: "7.7",
    kurtosis: "23",
    rows: [
      { expected: "activate", prompt: "### Section 1: Discovery ... ### Section 2: Analysis", target: "2", score: "-11.160", rank: "1" },
      { expected: "activate", prompt: "Cat: maomao. Dog: wowowow? Cat: mamoa.", target: "Cat", score: "-7.496", rank: "2" },
      { expected: "not activate", prompt: "### Discovery ... ### Analysis", target: "Analysis", score: "+0.352", rank: "502" },
      { expected: "not activate", prompt: "At 3:30 p.m., the race began, and the score was 3:2 after the first round.", target: "3", score: "-1.737", rank: "55" },
    ],
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

function saeExplorerUrl(item) {
  const params = new URLSearchParams({
    model: item.model,
    model_name: item.model,
    layer: item.layer,
    text: item.text,
    probe_text: item.text,
    top_k: String(item.topK || 5),
    card_width: String(item.cardWidth || 140),
  });

  if (item.saeFeature) {
    params.set("features", String(item.saeFeature));
    params.set("selected_features", String(item.saeFeature));
  }

  return `${saeExplorerBase}?${params.toString()}`;
}


function highlightTarget(prompt, target) {
  const text = String(prompt);
  const needle = String(target || "");
  if (!needle) return escapeHtml(text);
  const index = text.indexOf(needle);
  if (index < 0) return escapeHtml(text);
  return `${escapeHtml(text.slice(0, index))}<mark>${escapeHtml(text.slice(index, index + needle.length))}</mark>${escapeHtml(text.slice(index + needle.length))}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function selectValidation(index) {
  const item = validationCases[index];
  document.querySelectorAll(".validation-tab").forEach((tab, tabIndex) => {
    tab.classList.toggle("active", tabIndex === index);
  });

  document.querySelector("#validation-meta").textContent = `${item.modelLabel} ${item.componentLabel} - ${item.confidence}, ${item.type}, ${item.side} side, ERF ${item.erf}, kappa ${item.kurtosis}`;
  document.querySelector("#validation-title").textContent = item.title;
  document.querySelector("#validation-prompts").innerHTML = item.rows.map((row) => {
    const rowUrl = explorerUrl({
      model: item.model,
      layer: item.layer,
      text: row.prompt,
      topK: 5,
      cardWidth: 150,
      opacityCutoff: 0.5,
      components: item.components,
    });

    return `
      <div class="prompt-row ${row.expected === "activate" ? "should-activate" : "should-not"}">
        <span class="expected">${escapeHtml(row.expected)}</span>
        <p title="${escapeHtml(row.prompt)}">${highlightTarget(row.prompt, row.target)}</p>
        <span class="score">${escapeHtml(row.score)}</span>
        <span class="rank">rank ${escapeHtml(row.rank)}</span>
        <a class="prompt-link hf-link" href="${escapeHtml(rowUrl)}" target="_blank" rel="noopener noreferrer"><svg aria-hidden="true" viewBox="0 0 24 24"><path d="M7 17 17 7"/><path d="M8 7h9v9"/><path d="M5 5v14h14"/></svg>Open</a>
      </div>
    `;
  }).join("");
}

function renderValidationCases() {
  const list = document.querySelector("#validation-tabs");
  if (!list) return;
  list.innerHTML = "";
  validationCases.forEach((item, index) => {
    const button = document.createElement("button");
    button.className = "validation-tab";
    button.type = "button";
    button.innerHTML = `
      <span>${escapeHtml(item.componentLabel)}</span>
      <strong>${escapeHtml(item.title)}</strong>
      <small>${escapeHtml(item.type)} - ${escapeHtml(item.side)} side</small>
    `;
    button.addEventListener("click", () => selectValidation(index));
    list.appendChild(button);
  });
  selectValidation(0);
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
  const bankUrl = explorerUrl(bankCase);
  document.querySelector("#hero-case-link").href = bankUrl;
  document.querySelector("#bank-inline-link").href = bankUrl;

  const overlapLinks = {
    "#overlap-gpt2-explorer": overlapExamples.gpt2,
    "#overlap-gemma-explorer": overlapExamples.gemma,
    "#overlap-qwen-explorer": overlapExamples.qwen,
  };
  const saeOverlapLinks = {
    "#overlap-qwen-sae": overlapExamples.qwen,
  };

  Object.entries(overlapLinks).forEach(([selector, item]) => {
    const link = document.querySelector(selector);
    if (link) link.href = explorerUrl(item);
  });
  Object.entries(saeOverlapLinks).forEach(([selector, item]) => {
    const link = document.querySelector(selector);
    if (link) link.href = saeExplorerUrl(item);
  });
}

function initSectionNav() {
  const links = Array.from(document.querySelectorAll("[data-section-nav]"));
  const sections = links
    .map((link) => document.getElementById(link.dataset.sectionNav))
    .filter(Boolean);

  if (!links.length || !sections.length) return;

  function setActive(id) {
    links.forEach((link) => {
      const isActive = link.dataset.sectionNav === id;
      link.classList.toggle("active", isActive);
      if (isActive) {
        link.setAttribute("aria-current", "true");
      } else {
        link.removeAttribute("aria-current");
      }
    });
  }

  function updateActiveSection() {
    const marker = window.innerHeight * 0.45;
    let active = sections[0];
    sections.forEach((section) => {
      if (section.getBoundingClientRect().top <= marker) {
        active = section;
      }
    });
    setActive(active.id);
  }

  links.forEach((link) => {
    link.addEventListener("click", () => setActive(link.dataset.sectionNav));
  });

  updateActiveSection();
  window.addEventListener("scroll", updateActiveSection, { passive: true });
  window.addEventListener("resize", updateActiveSection);
}

renderCases();
renderValidationCases();
initLinks();
initSectionNav();
selectCase(0);
