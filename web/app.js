"use strict";
(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const types = { paper: "论文", blog: "博客", doc: "文档" };
  const statuses = { unread: "未读", reading: "在读", read: "已读" };
  const dimensions = {
    topic: "研究主题",
    architecture: "模型架构",
    transform: "变换",
    availability: "代码可用性",
    publication: "出版状态",
    target: "量化对象",
    scheme: "技术方案",
    granularity: "量化粒度",
    precision: "数值精度",
    method: "具体方法",
    system: "系统优化",
    workload: "工作负载",
    stage: "推理阶段",
    memory: "内存架构",
    hardware: "目标硬件",
    backend: "运行后端",
    evidence: "实现证据",
  };
  const state = {
    token: "",
    tags: {},
    all: [],
    selected: new Set(),
    detail: null,
    editDirty: false,
    importDirty: false,
    listRequest: 0,
    detailRequest: 0,
  };
  const detailDialog = $("#detail-dialog"),
    importDialog = $("#import-dialog");
  let searchTimer,
    toastTimer,
    diagramSequence = 0;
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text !== undefined && text !== null) element.textContent = String(text);
    if (className) element.className = className;
    return element;
  };
  function button(text, action, className) {
    const element = node("button", text, className);
    element.type = "button";
    element.addEventListener("click", action);
    return element;
  }
  function message(target, text, kind = "") {
    target.replaceChildren();
    if (text) target.append(node("div", text, `notice ${kind}`));
  }
  function toast(text) {
    const element = $("#toast");
    element.textContent = text;
    element.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => {
      element.hidden = true;
    }, 5000);
  }
  function safeLink(label, url) {
    try {
      const parsed = new URL(url, location.origin);
      if (!["http:", "https:"].includes(parsed.protocol))
        return node("span", label);
      const a = node("a", label);
      a.href = parsed.href;
      a.target = "_blank";
      a.rel = "noopener noreferrer";
      return a;
    } catch {
      return node("span", label);
    }
  }
  async function api(path, options = {}) {
    const response = await fetch(path, {
      credentials: "same-origin",
      ...options,
      headers: {
        "Content-Type": "application/json",
        ...(options.method ? { "X-Catalog-Token": state.token } : {}),
        ...options.headers,
      },
    });
    let data;
    try {
      data = await response.json();
    } catch {
      throw new Error(`服务器返回了无法读取的响应（${response.status}）。`);
    }
    if (!response.ok) {
      const error = new Error(data.error || `请求失败（${response.status}）。`);
      error.status = response.status;
      error.duplicates = data.duplicates;
      throw error;
    }
    return data;
  }
  function tagLabel(key) {
    return state.tags[key]?.label || key;
  }
  function renderVocabulary() {
    const query = $("#tag-search").value.trim().toLocaleLowerCase();
    const groups = new Map();
    const counts = new Map();
    state.all.forEach((entry) =>
      (entry.tags || []).forEach((key) =>
        counts.set(key, (counts.get(key) || 0) + 1),
      ),
    );
    Object.entries(state.tags).forEach(([key, value]) => {
      if (
        query &&
        ![key, value.label, ...(value.aliases || [])]
          .join(" ")
          .toLocaleLowerCase()
          .includes(query)
      )
        return;
      const dimension = key.split("/")[0];
      if (!groups.has(dimension)) groups.set(dimension, []);
      groups.get(dimension).push([key, value]);
    });
    const root = $("#tag-vocabulary");
    root.replaceChildren();
    for (const [dimension, tags] of groups) {
      const group = node("details", null, "tag-group");
      group.open = true;
      group.append(
        node(
          "summary",
          `${dimensions[dimension] || dimension} · ${tags.length}`,
        ),
      );
      tags.forEach(([key, value]) => {
        const label = node("label", null, "tag-option");
        const input = node("input");
        input.type = "checkbox";
        input.value = key;
        input.checked = state.selected.has(key);
        input.addEventListener("change", () => {
          input.checked ? state.selected.add(key) : state.selected.delete(key);
          renderFilters();
          loadList();
        });
        const text = node("span", value.label || key);
        text.append(node("small", key));
        label.append(
          input,
          text,
          node("span", counts.get(key) || 0, "tag-count"),
        );
        group.append(label);
      });
      root.append(group);
    }
    if (!groups.size) root.append(node("p", "没有匹配的标签。", "muted small"));
  }
  function renderFilters() {
    const root = $("#selected-filters");
    root.replaceChildren();
    state.selected.forEach((key) =>
      root.append(
        button(
          `${tagLabel(key)} ×`,
          () => {
            state.selected.delete(key);
            renderVocabulary();
            renderFilters();
            loadList();
          },
          "chip",
        ),
      ),
    );
    if ($("#query").value.trim())
      root.append(node("span", `关键词：${$("#query").value.trim()}`, "chip"));
    const kind = $("#filter-type").value,
      status = $("#filter-status").value;
    if (kind) root.append(node("span", types[kind], "chip"));
    if (status) root.append(node("span", statuses[status], "chip"));
  }
  async function loadList() {
    const request = ++state.listRequest;
    const params = new URLSearchParams();
    if ($("#query").value.trim()) params.set("q", $("#query").value.trim());
    if ($("#filter-type").value) params.set("type", $("#filter-type").value);
    if ($("#filter-status").value)
      params.set("status", $("#filter-status").value);
    state.selected.forEach((key) => params.append("tag", key));
    $("#resource-list").classList.add("loading");
    $("#resource-list").setAttribute("aria-busy", "true");
    message($("#catalog-message"), "正在读取资源目录…");
    try {
      const data = await api(`/api/catalog?${params}`);
      if (request !== state.listRequest) return;
      state.token = data.token;
      state.tags = data.tags;
      renderList(data.entries);
      $("#result-count").textContent = `${data.total} 条`;
      message($("#catalog-message"), "");
    } catch (error) {
      if (request === state.listRequest)
        message($("#catalog-message"), error.message, "error");
    } finally {
      if (request === state.listRequest) {
        $("#resource-list").classList.remove("loading");
        $("#resource-list").removeAttribute("aria-busy");
      }
    }
  }
  function renderList(entries) {
    const root = $("#resource-list");
    root.replaceChildren();
    if (!entries.length) {
      const empty = node("div", null, "empty-state");
      empty.append(
        node("h3", "这个交集里，还没有资源。"),
        node("p", "试试减少标签或清除关键词；也可以收录新的研究线索。"),
      );
      root.append(empty);
      return;
    }
    for (const entry of entries) {
      const card = node("article", null, "resource-card");
      card.dataset.resourceId = entry.id;
      const content = node("div"),
        meta = node("div", null, "card-meta");
      meta.append(node("span", types[entry.type] || entry.type, "type-badge"));
      if (entry.year) meta.append(node("span", entry.year));
      if (entry.venue) meta.append(node("span", entry.venue));
      if (entry.notes_path)
        meta.append(node("span", "有研究笔记", "note-badge"));
      const title = node("a", entry.title, "card-title");
      title.href = `#resource=${encodeURIComponent(entry.id)}`;
      title.addEventListener("click", (event) => {
        event.preventDefault();
        selectResource(entry.id);
      });
      content.append(
        meta,
        title,
        node("p", entry.summary || "尚未填写收藏理由。", "card-summary"),
      );
      const tags = node("div", null, "tag-line");
      (entry.tags || []).forEach((key) =>
        tags.append(node("span", tagLabel(key), "tag-label")),
      );
      content.append(tags);
      const side = node("div", null, "card-side");
      side.append(
        node(
          "span",
          statuses[entry.status] || entry.status,
          `status status-${entry.status}`,
        ),
        node("span", entry.id, "card-id"),
      );
      card.append(content, side);
      root.append(card);
    }
  }
  async function refreshCatalog() {
    const data = await api("/api/catalog");
    state.all = data.entries;
    state.tags = data.tags;
    state.token = data.token;
    $("#library-count").textContent = data.total;
    renderVocabulary();
    await loadList();
  }
  function discardChanges(kind) {
    const dirty = kind === "edit" ? state.editDirty : state.importDirty;
    return !dirty || window.confirm("有尚未保存的修改，确定放弃吗？");
  }
  function setHash(id, replace = false, tab = null) {
    const url = new URL(location.href);
    const params = new URLSearchParams();
    if (id) params.set("resource", id);
    if (id && tab) params.set("tab", tab);
    url.hash = params.toString();
    history[replace ? "replaceState" : "pushState"](null, "", url);
  }
  async function selectResource(id, fromHash = false) {
    if (!discardChanges("edit")) {
      if (fromHash && state.detail) setHash(state.detail.entry.id, true);
      return;
    }
    state.editDirty = false;
    if (!fromHash) setHash(id);
    if (!detailDialog.open) detailDialog.showModal();
    const loadingTitle = node("h1", "资源详情", "detail-title");
    loadingTitle.id = "detail-title";
    $("#detail-content").replaceChildren(loadingTitle);
    message($("#detail-message"), "正在读取资源与笔记…");
    const request = ++state.detailRequest;
    try {
      const data = await api(`/api/resources/${encodeURIComponent(id)}`);
      if (request !== state.detailRequest || !detailDialog.open) return;
      state.detail = data;
      message($("#detail-message"), "");
      renderDetail();
    } catch (error) {
      if (request === state.detailRequest)
        message($("#detail-message"), error.message, "error");
    }
  }
  function closeDetail() {
    if (!discardChanges("edit")) return;
    state.editDirty = false;
    ++state.detailRequest;
    detailDialog.close();
    setHash(null);
  }
  function renderDetail() {
    const data = state.detail,
      entry = data.entry,
      root = $("#detail-content");
    root.replaceChildren();
    const meta = node("div", null, "card-meta");
    meta.append(
      node("span", types[entry.type] || entry.type, "type-badge"),
      node("span", statuses[entry.status] || entry.status),
      node("span", entry.id),
    );
    const title = node("h1", entry.title, "detail-title");
    title.id = "detail-title";
    root.append(
      meta,
      title,
      node("p", entry.summary || "尚未填写收藏理由。", "detail-summary"),
    );
    const tags = node("div", null, "tag-line");
    (entry.tags || []).forEach((key) =>
      tags.append(node("span", `${tagLabel(key)} · ${key}`, "tag-label")),
    );
    root.append(tags);
    const toolbar = node("div", null, "detail-toolbar");
    toolbar.append(
      safeLink("访问原始来源 ↗", entry.url),
      button("编辑元数据", () => {
        if (!$("#detail-editor")) {
          const editor = makeEditor(entry, false);
          editor.id = "detail-editor";
          root.append(editor);
          editor.scrollIntoView({ block: "start", behavior: "smooth" });
          $("input", editor)?.focus();
        }
      }),
      button("重新读取文件", () => selectResource(entry.id)),
    );
    root.append(toolbar);
    const fields = node("dl", null, "metadata-grid");
    const display = {
      authors: "作者",
      year: "年份",
      venue: "发表来源",
      doi: "DOI",
      metadata_path: "元数据路径",
      notes_path: "笔记路径",
    };
    Object.entries(display).forEach(([key, label]) => {
      if (entry[key] && (!Array.isArray(entry[key]) || entry[key].length))
        fields.append(
          node("dt", label),
          node(
            "dd",
            Array.isArray(entry[key]) ? entry[key].join("；") : entry[key],
          ),
        );
    });
    if (entry.links?.code) {
      const repository = node("dd");
      repository.append(safeLink(entry.links.code, entry.links.code));
      fields.append(node("dt", "代码仓库"), repository);
    }
    const relatedLinks = Object.entries(entry.links || {}).filter(
      ([name]) => name !== "code",
    );
    if (relatedLinks.length) {
      const links = node("dd");
      relatedLinks.forEach(([name, url]) => {
        links.append(safeLink(name, url), document.createTextNode("  "));
      });
      fields.append(node("dt", "相关链接"), links);
    }
    if (entry.attachments?.length) {
      const files = node("dd");
      entry.attachments.forEach((file) => {
        const path = String(file.path || "");
        if (path.startsWith("resources/") && !path.split("/").includes(".."))
          files.append(
            safeLink(
              path,
              `/files/${encodeURIComponent(entry.id)}/${path.split("/").map(encodeURIComponent).join("/")}`,
            ),
          );
        files.append(document.createTextNode("  "));
      });
      fields.append(node("dt", "本地附件"), files);
    }
    root.append(fields);
    const tabs = node("div", null, "tabs");
    tabs.setAttribute("role", "tablist");
    tabs.setAttribute("aria-label", "资源内容");
    const body = node("section", null, "prose");
    body.id = "resource-body";
    body.setAttribute("role", "tabpanel");
    body.tabIndex = 0;
    const notes = node("section", null, "prose");
    notes.id = "resource-notes";
    notes.setAttribute("role", "tabpanel");
    notes.tabIndex = 0;
    notes.hidden = true;
    // Only these two fragments originate from the backend's safe Markdown renderer.
    body.innerHTML = data.body_html || "";
    if (!body.textContent.trim() && !body.children.length)
      body.append(node("p", "这张资源卡尚无正文。", "muted"));
    if (data.notes_html) notes.innerHTML = data.notes_html;
    else
      notes.append(
        node(
          "p",
          "尚无研究笔记。在仓库对应目录中创建 notes.md 后，点击「重新读取文件」查看。",
          "muted",
        ),
      );
    notes.prepend(
      node(
        "p",
        "笔记为只读视图；公式使用本地 KaTeX 排版，修改请使用仓库中的 Markdown 文件。",
        "readonly-note",
      ),
    );
    const panels = [body, notes];
    [
      "资源正文",
      entry.notes_path ? "研究笔记 · 已记录" : "研究笔记 · 未创建",
    ].forEach((label, index) => {
      const tab = button(label, () => activateTab(index));
      tab.id = `detail-tab-${index}`;
      tab.setAttribute("role", "tab");
      tab.setAttribute("aria-controls", panels[index].id);
      tab.setAttribute("aria-selected", String(index === 0));
      tab.tabIndex = index === 0 ? 0 : -1;
      panels[index].setAttribute("aria-labelledby", tab.id);
      tabs.append(tab);
      tab.addEventListener("keydown", (event) => {
        if (["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
          event.preventDefault();
          const target =
            event.key === "Home" ? 0 : event.key === "End" ? 1 : 1 - index;
          activateTab(target);
          tabs.children[target].focus();
        }
      });
    });
    function activateTab(index) {
      panels.forEach((panel, i) => {
        panel.hidden = i !== index;
        tabs.children[i].setAttribute("aria-selected", String(i === index));
        tabs.children[i].tabIndex = i === index ? 0 : -1;
      });
      setHash(entry.id, true, index === 1 ? "notes" : null);
      renderMath(panels[index]);
      renderMermaid(panels[index]);
    }
    root.append(tabs, body, notes);
    activateTab(new URLSearchParams(location.hash.slice(1)).get("tab") === "notes" ? 1 : 0);
    const known = new Set([
      "id",
      "type",
      "title",
      "url",
      "status",
      "summary",
      "tags",
      "authors",
      "year",
      "venue",
      "doi",
      "links",
      "attachments",
      "metadata_path",
      "notes_path",
    ]);
    const extras = Object.fromEntries(
      Object.entries(entry).filter(([key]) => !known.has(key)),
    );
    if (Object.keys(extras).length) {
      const details = node("details");
      details.append(
        node("summary", "其他公开元数据"),
        node("pre", JSON.stringify(extras, null, 2)),
      );
      root.append(details);
    }
  }
  function renderMath(root) {
    const formulas = root.querySelectorAll(".math-source:not([data-math-rendered])");
    if (!formulas.length) return;
    if (!window.katex) {
      if (!$(".math-warning", root))
        root.prepend(node("p", "本地 KaTeX 组件未加载，公式保留为原始 LaTeX。", "notice error math-warning"));
      return;
    }
    for (const element of formulas) {
      const source = element.textContent;
      try {
        window.katex.render(source, element, {
          displayMode: element.dataset.mathDisplay === "block",
          output: "htmlAndMathml",
          trust: false,
          throwOnError: true,
          maxExpand: 1000,
          maxSize: 20,
          macros: {},
        });
      } catch (error) {
        element.classList.add("math-error");
        element.replaceChildren(
          node("span", "公式无法渲染："),
          node("code", source),
        );
        element.title = error.message;
      }
      element.dataset.mathRendered = "true";
    }
  }
  let mermaidReady = false;
  async function renderMermaid(root) {
    const blocks = [...root.querySelectorAll("code.language-mermaid")];
    if (!blocks.length) return;
    if (!window.mermaid) {
      if (!$(".mermaid-warning", root))
        root.append(
          node(
            "p",
            "本地 Mermaid 组件不可用；图表保留为源代码。",
            "notice warning mermaid-warning",
          ),
        );
      return;
    }
    if (!mermaidReady) {
      window.mermaid.initialize({
        securityLevel: "strict",
        startOnLoad: false,
        flowchart: { htmlLabels: false },
      });
      mermaidReady = true;
    }
    for (const code of blocks) {
      if (code.dataset.rendering) continue;
      code.dataset.rendering = "true";
      try {
        const result = await window.mermaid.render(
          `research-diagram-${++diagramSequence}`,
          code.textContent,
        );
        if (!code.isConnected) continue;
        const figure = node("figure", null, "mermaid-diagram");
        figure.innerHTML = result.svg;
        code.parentElement.replaceWith(figure);
      } catch {
        if (code.isConnected)
          code.parentElement.after(
            node(
              "p",
              "Mermaid 图表无法解析，已保留原始代码。",
              "notice warning",
            ),
          );
      }
    }
  }
  const requiredKeys = new Set([
    "id",
    "type",
    "title",
    "url",
    "status",
    "summary",
    "tags",
  ]);
  const primaryKeys = new Set([
    ...requiredKeys,
    "authors",
    "year",
    "venue",
    "doi",
    "links",
    "attachments",
    "metadata_path",
    "notes_path",
  ]);
  function makeEditor(metadata, creating) {
    const original = structuredClone(metadata);
    const form = node("form", null, "editor");
    form.dataset.mode = creating ? "create" : "edit";
    form.append(
      node("h3", creating ? "确认收录信息" : "编辑资源元数据"),
      node(
        "p",
        creating
          ? "请确认 ID、类型与中文收藏理由；只有显式选择的标签会被写入。"
          : "仅提交你修改的字段。正文、笔记、未知字段及未修改的数据类型保持不变。ID 与类型不可更改。",
        "readonly-note",
      ),
    );
    const grid = node("div", null, "form-grid");
    const controls = new Map(),
      initialValues = new Map();
    function field(key, label, options = {}) {
      const wrapper = node(
        "label",
        null,
        `field${options.wide ? " wide" : ""}`,
      );
      wrapper.append(document.createTextNode(label));
      let input;
      if (options.choices) {
        input = node("select");
        Object.entries(options.choices).forEach(([value, text]) => {
          const option = node("option", text);
          option.value = value;
          input.append(option);
        });
      } else {
        input = node(options.area ? "textarea" : "input");
        if (!options.area) input.type = options.type || "text";
      }
      input.name = key;
      input.id = `${creating ? "create" : "edit"}-${key}`;
      if (options.required) input.required = true;
      if (options.readonly) {
        if (input.tagName === "SELECT") input.disabled = true;
        else input.readOnly = true;
      }
      let value = key === "links.code" ? original.links?.code : original[key];
      if (key === "links" && value)
        value = Object.fromEntries(
          Object.entries(value).filter(([name]) => name !== "code"),
        );
      if (options.json) {
        input.classList.add("json-field");
        value = value === undefined ? "" : JSON.stringify(value, null, 2);
      } else if (key === "authors")
        value = Array.isArray(value) ? value.join("\n") : "";
      input.value =
        value ?? (key === "status" ? "unread" : key === "type" ? "paper" : "");
      if (key === "id") input.pattern = "[a-z0-9]+(?:-[a-z0-9]+)*";
      if (key === "year") {
        input.min = "1";
        input.max = "9999";
        input.step = "1";
      }
      wrapper.append(input);
      if (options.hint) wrapper.append(node("small", options.hint));
      controls.set(key, input);
      initialValues.set(key, input.value);
      return wrapper;
    }
    grid.append(
      field("title", "标题 *", { required: true, wide: true }),
      field("id", "唯一 ID *", {
        required: true,
        readonly: !creating,
        hint: "小写英文、数字、连字符；保存后不可更改。",
      }),
      field("type", "资源类型 *", { choices: types, readonly: !creating }),
      field("url", "原始网址 *", { required: true, type: "url", wide: true }),
      field("links.code", "代码仓库", {
        type: "url",
        wide: true,
        hint: "已核实的代码仓库 HTTP(S) 网址；可选，留空移除，不自动标记开源。",
      }),
      field("summary", "中文收藏理由 / 摘要", {
        area: true,
        wide: true,
        hint: "写下与你的研究有关的内容；不会自动复制来源摘要。",
      }),
      field("status", "阅读状态", { choices: statuses }),
      field("year", "年份", { type: "number" }),
      field("authors", "作者", {
        area: true,
        hint: "每行一位作者；留空可移除此可选字段。",
      }),
      field("venue", "发表来源 / 会议"),
      field("doi", "DOI", { wide: true }),
    );
    const tagField = node("fieldset", null, "field wide");
    tagField.style.border = "0";
    tagField.style.padding = "0";
    tagField.style.margin = "0";
    tagField.append(node("legend", "规范标签（可多选）"));
    const tagBox = node("div", null, "edit-tags");
    const tagInputs = [];
    Object.entries(state.tags).forEach(([key, value]) => {
      const label = node("label");
      const check = node("input");
      check.type = "checkbox";
      check.name = "tags";
      check.value = key;
      check.checked = (original.tags || []).includes(key);
      const text = node("span", value.label || key);
      text.append(node("small", key));
      label.append(check, text);
      tagBox.append(label);
      tagInputs.push(check);
    });
    tagField.append(tagBox);
    grid.append(tagField);
    const advanced = node("details", null, "advanced");
    advanced.append(node("summary", "高级字段：相关链接、附件与自定义元数据"));
    advanced.append(
      field("links", "其他相关链接（JSON 对象）", {
        area: true,
        json: true,
        hint: '例如 {"project": "https://…"}；不含 code，代码仓库使用上方输入框。留空移除其他链接。',
      }),
      field("attachments", "附件（JSON 数组）", {
        area: true,
        json: true,
        hint: "保留已有 resources/ 相对路径；此处不下载附件。留空移除。",
      }),
    );
    const extras = Object.fromEntries(
      Object.entries(original).filter(([key]) => !primaryKeys.has(key)),
    );
    const extraField = field("custom", "自定义公开字段（JSON 对象）", {
      area: true,
      json: true,
      hint: "未修改时原样保留。删除对象中的键表示明确移除此自定义字段；不要填写系统或必填字段。",
    });
    controls.get("custom").value = JSON.stringify(extras, null, 2);
    initialValues.set("custom", controls.get("custom").value);
    advanced.append(extraField);
    grid.append(advanced);
    form.append(grid);
    const feedback = node("div");
    feedback.setAttribute("role", "status");
    feedback.setAttribute("aria-live", "polite");
    form.append(feedback);
    const actions = node("div", null, "form-actions"),
      submit = node(
        "button",
        creating ? "确认收录到仓库" : "保存元数据",
        "primary",
      );
    submit.type = "submit";
    actions.append(submit);
    if (!creating)
      actions.append(
        button("取消编辑", () => {
          if (discardChanges("edit")) {
            state.editDirty = false;
            form.remove();
          }
        }),
      );
    form.append(actions);
    form.addEventListener("input", () => {
      state[creating ? "importDirty" : "editDirty"] = true;
    });
    form.addEventListener("change", () => {
      state[creating ? "importDirty" : "editDirty"] = true;
    });
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      message(feedback, "");
      let updates = {},
        removeFields = [];
      try {
        for (const [key, input] of controls) {
          if (
            ["custom", "links", "links.code"].includes(key) ||
            (!creating && ["id", "type"].includes(key))
          )
            continue;
          if (!creating && input.value === initialValues.get(key)) continue;
          const value = input.value;
          if (!requiredKeys.has(key) && !value.trim()) {
            if (!creating && Object.hasOwn(original, key))
              removeFields.push(key);
            continue;
          }
          if (key === "authors")
            updates[key] = value
              .split(/\r?\n/)
              .map((item) => item.trim())
              .filter(Boolean);
          else if (key === "year") updates[key] = Number(value);
          else if (key === "attachments")
            updates[key] = JSON.parse(value);
          else updates[key] = value;
        }
        const codeInput = controls.get("links.code"),
          linksInput = controls.get("links");
        if (
          creating ||
          codeInput.value !== initialValues.get("links.code") ||
          linksInput.value !== initialValues.get("links")
        ) {
          const otherLinks = linksInput.value.trim()
            ? JSON.parse(linksInput.value)
            : {};
          if (
            !otherLinks ||
            Array.isArray(otherLinks) ||
            typeof otherLinks !== "object"
          )
            throw new Error("其他相关链接必须是 JSON 对象。");
          if (Object.hasOwn(otherLinks, "code"))
            throw new Error("请在“代码仓库”输入框填写代码网址，不在其他相关链接中填写 code。");
          const links = { ...otherLinks };
          const code = codeInput.value.trim();
          if (code) links.code = code;
          if (Object.keys(links).length) updates.links = links;
          else if (!creating && Object.hasOwn(original, "links"))
            removeFields.push("links");
        }
        const tags = tagInputs
          .filter((input) => input.checked)
          .map((input) => input.value);
        if (
          creating ||
          JSON.stringify([...tags].sort()) !==
            JSON.stringify([...(original.tags || [])].sort())
        )
          updates.tags = tags;
        if (
          creating ||
          controls.get("custom").value !== initialValues.get("custom")
        ) {
          const custom = JSON.parse(controls.get("custom").value);
          if (!custom || Array.isArray(custom) || typeof custom !== "object")
            throw new Error("自定义字段必须是 JSON 对象。");
          for (const [key, value] of Object.entries(custom)) {
            if (
              primaryKeys.has(key) ||
              ["__proto__", "constructor", "prototype", "zotero"].includes(key)
            )
              throw new Error(`自定义字段不能使用保留键：${key}`);
            if (
              creating ||
              !Object.hasOwn(extras, key) ||
              JSON.stringify(value) !== JSON.stringify(extras[key])
            )
              Object.defineProperty(updates, key, {
                value,
                enumerable: true,
                configurable: true,
              });
          }
          for (const key of Object.keys(extras))
            if (!Object.hasOwn(custom, key)) removeFields.push(key);
        }
      } catch (error) {
        message(feedback, `无法提交：${error.message}`, "error");
        return;
      }
      submit.disabled = true;
      try {
        const data = creating
          ? await api("/api/resources", {
              method: "POST",
              body: JSON.stringify({ metadata: updates }),
            })
          : await api(`/api/resources/${encodeURIComponent(original.id)}`, {
              method: "PUT",
              body: JSON.stringify({
                revision: state.detail.revision,
                updates,
                remove_fields: removeFields,
              }),
            });
        state[creating ? "importDirty" : "editDirty"] = false;
        if (creating) {
          closeImport();
          setHash(data.entry.id);
          if (!detailDialog.open) detailDialog.showModal();
        }
        state.detail = data;
        message($("#detail-message"), "");
        renderDetail();
        toast(
          creating
            ? "资源已收录；未下载全文或创建笔记。"
            : "已保存元数据并更新仓库索引。",
        );
        try {
          await refreshCatalog();
        } catch (error) {
          message(
            $("#catalog-message"),
            `保存已成功，但目录刷新失败：${error.message}`,
            "error",
          );
        }
      } catch (error) {
        message(
          feedback,
          error.status === 409 && !creating
            ? `保存冲突：${error.message} 当前输入已保留。请先复制你的修改，再重新读取文件后合并；不会覆盖外部更改。`
            : error.message,
          "error",
        );
        if (error.duplicates?.length)
          renderDuplicates(feedback, error.duplicates);
        if (error.status === 409 && !creating)
          feedback.append(
            button("放弃本地编辑并重新读取", () => selectResource(original.id)),
          );
      } finally {
        submit.disabled = false;
      }
    });
    return form;
  }
  function renderDuplicates(target, duplicates) {
    if (!duplicates?.length) return;
    const panel = node("div", null, "duplicates");
    panel.append(node("h3", "发现可能已收录的资源，请先核对"));
    duplicates.forEach((item) =>
      panel.append(
        button(
          `${item.title} · ${item.reason || "可能重复"} · ${item.metadata_path || item.id}`,
          () => {
            if (!discardChanges("import") || !discardChanges("edit")) return;
            state.importDirty = false;
            state.editDirty = false;
            closeImport();
            selectResource(item.id);
          },
        ),
      ),
    );
    target.append(panel);
  }
  function beginImport(metadata, manual = false) {
    const root = $("#import-editor");
    root.replaceChildren();
    if (manual)
      root.append(
        node(
          "div",
          "手动补录 · 以下元数据未经来源核实，请自行检查标题、作者、年份和版本。",
          "notice warning",
        ),
      );
    root.append(makeEditor(metadata, true));
    state.importDirty = true;
  }
  $("#open-import").addEventListener("click", () => {
    if (!discardChanges("edit")) return;
    if (state.editDirty) {
      state.editDirty = false;
      renderDetail();
    }
    if (!importDialog.open) importDialog.showModal();
    $("#import-url").focus();
  });
  function closeImport() {
    if (discardChanges("import")) {
      ++previewRequest;
      state.importDirty = false;
      importDialog.close();
      $("#import-editor").replaceChildren();
      $("#import-preview").replaceChildren();
      message($("#import-message"), "");
    }
  }
  $("#close-import").addEventListener("click", closeImport);
  importDialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeImport();
  });
  $("#close-detail").addEventListener("click", closeDetail);
  detailDialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeDetail();
  });
  $("#import-url").addEventListener("input", () => {
    state.importDirty = true;
  });
  $("#manual-import").addEventListener("click", () => {
    if ($("#import-editor").children.length && !discardChanges("import"))
      return;
    ++previewRequest;
    message($("#import-message"), "");
    $("#import-preview").replaceChildren();
    beginImport(
      {
        id: "",
        title: "",
        type: "paper",
        url: $("#import-url").value,
        status: "unread",
        summary: "",
        tags: [],
      },
      true,
    );
  });
  let previewRequest = 0;
  $("#preview-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if ($("#import-editor").children.length && !discardChanges("import"))
      return;
    const request = ++previewRequest,
      control = $("#preview-button");
    control.disabled = true;
    $("#import-preview").replaceChildren();
    $("#import-editor").replaceChildren();
    message($("#import-message"), "正在读取来源；预览不会写入仓库…");
    try {
      const data = await api("/api/import/preview", {
        method: "POST",
        body: JSON.stringify({ url: $("#import-url").value }),
      });
      if (request !== previewRequest || !importDialog.open) return;
      message($("#import-message"), "已取得来源预览，请确认后再收录。");
      const root = $("#import-preview"),
        source = node("div", null, "preview-source");
      source.append(
        node("h3", "来源摘要 · 不会自动写入中文收藏理由"),
        node("p", data.description || "来源未提供可读取的摘要。"),
        safeLink("查看核实来源 ↗", data.source_url),
      );
      root.append(source);
      (data.warnings || []).forEach((warning) =>
        root.append(node("div", warning, "notice warning")),
      );
      renderDuplicates(root, data.duplicates);
      beginImport(data.metadata);
    } catch (error) {
      if (request === previewRequest && importDialog.open) {
        message(
          $("#import-message"),
          `${error.message} 网址已保留。可以重试，或点击上方「直接手动填写」继续，手动信息将明确标记为未经来源核实。`,
          "error",
        );
        if (error.duplicates)
          renderDuplicates($("#import-preview"), error.duplicates);
      }
    } finally {
      control.disabled = false;
    }
  });
  $("#filter-form").addEventListener("submit", (event) => {
    event.preventDefault();
    clearTimeout(searchTimer);
    renderFilters();
    loadList();
  });
  $("#query").addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      renderFilters();
      loadList();
    }, 250);
  });
  ["#filter-type", "#filter-status"].forEach((selector) =>
    $(selector).addEventListener("change", () => {
      renderFilters();
      loadList();
    }),
  );
  $("#tag-search").addEventListener("input", renderVocabulary);
  $("#reset-filters").addEventListener("click", () => {
    clearTimeout(searchTimer);
    $("#filter-form").reset();
    state.selected.clear();
    renderFilters();
    renderVocabulary();
    loadList();
  });
  $("#refresh").addEventListener("click", async (event) => {
    const control = event.currentTarget;
    control.disabled = true;
    try {
      await refreshCatalog();
      toast("已重新读取仓库目录。打开的资源可通过「重新读取文件」更新。");
    } catch (error) {
      message($("#catalog-message"), error.message, "error");
    } finally {
      control.disabled = false;
    }
  });
  function followHash() {
    const params = new URLSearchParams(location.hash.slice(1));
    const id = params.get("resource");
    if (id) {
      if (state.detail?.entry.id !== id || !detailDialog.open)
        selectResource(id, true);
      else $("#detail-tab-" + (params.get("tab") === "notes" ? 1 : 0))?.click();
    } else if (detailDialog.open) {
      if (discardChanges("edit")) {
        state.editDirty = false;
        detailDialog.close();
        ++state.detailRequest;
      } else if (state.detail) setHash(state.detail.entry.id, true);
    }
  }
  window.addEventListener("hashchange", followHash);
  window.addEventListener("popstate", followHash);
  window.addEventListener("beforeunload", (event) => {
    if (state.editDirty || state.importDirty) {
      event.preventDefault();
      event.returnValue = "";
    }
  });
  async function start() {
    message($("#catalog-message"), "正在连接本地资源仓库…");
    try {
      await refreshCatalog();
      followHash();
    } catch (error) {
      message(
        $("#catalog-message"),
        `${error.message} 请确认本地服务正在运行，再点击「刷新仓库」。`,
        "error",
      );
    }
  }
  start();
})();
