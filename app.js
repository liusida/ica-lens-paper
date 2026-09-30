function initSectionNav() {
  const links = Array.from(document.querySelectorAll("[data-section-nav]"));
  const sections = links.map((link) => document.getElementById(link.dataset.sectionNav)).filter(Boolean);
  if (!links.length || !sections.length) return;

  function setActive(id) {
    links.forEach((link) => {
      const active = link.dataset.sectionNav === id;
      link.classList.toggle("active", active);
      if (active) link.setAttribute("aria-current", "true");
      else link.removeAttribute("aria-current");
    });
  }

  function update() {
    const marker = window.innerHeight * 0.45;
    let active = sections[0];
    sections.forEach((section) => {
      if (section.getBoundingClientRect().top <= marker) active = section;
    });
    setActive(active.id);
  }

  links.forEach((link) => link.addEventListener("click", () => setActive(link.dataset.sectionNav)));
  window.addEventListener("scroll", update, { passive: true });
  window.addEventListener("resize", update);
  update();
}

function initEvidenceTabs() {
  const tabs = Array.from(document.querySelectorAll("[data-evidence]"));
  const panels = Array.from(document.querySelectorAll("[data-panel]"));
  function select(tab) {
    const selected = tab.dataset.evidence;
    tabs.forEach((item) => {
      const active = item === tab;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", String(active));
      item.tabIndex = active ? 0 : -1;
    });
    panels.forEach((panel) => {
      const active = panel.dataset.panel === selected;
      panel.classList.toggle("active", active);
      panel.hidden = !active;
    });
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const offset = event.key === "ArrowRight" ? 1 : -1;
      const next = tabs[(index + offset + tabs.length) % tabs.length];
      select(next);
      next.focus();
    });
  });
}

function initAutoTabs() {
  const tabs = Array.from(document.querySelectorAll("[data-auto]"));
  const panels = Array.from(document.querySelectorAll("[data-auto-panel]"));
  if (!tabs.length) return;

  function select(tab) {
    const selected = tab.dataset.auto;
    tabs.forEach((item) => {
      const active = item === tab;
      item.classList.toggle("active", active);
      item.setAttribute("aria-selected", String(active));
      item.tabIndex = active ? 0 : -1;
    });
    panels.forEach((panel) => {
      const active = panel.dataset.autoPanel === selected;
      panel.classList.toggle("active", active);
      panel.hidden = !active;
    });
  }

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
      event.preventDefault();
      const offset = event.key === "ArrowRight" ? 1 : -1;
      const next = tabs[(index + offset + tabs.length) % tabs.length];
      select(next);
      next.focus();
    });
  });
}

function initCopyButtons() {
  document.querySelectorAll("[data-copy-target]").forEach((button) => {
    button.addEventListener("click", async () => {
      const target = document.getElementById(button.dataset.copyTarget);
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.innerText);
        const original = button.textContent;
        button.textContent = "Copied";
        window.setTimeout(() => { button.textContent = original; }, 1400);
      } catch {
        button.textContent = "Select text";
      }
    });
  });
}

initSectionNav();
initEvidenceTabs();
initAutoTabs();
initCopyButtons();
