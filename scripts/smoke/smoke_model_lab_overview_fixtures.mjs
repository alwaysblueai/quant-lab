import { mkdir, rm, writeFile } from "node:fs/promises";
import { spawn } from "node:child_process";
import { join, resolve } from "node:path";

const BASE_URL = process.env.MODEL_LAB_SMOKE_BASE_URL || "http://127.0.0.1:8766";
const OUT_DIR = resolve(
  process.env.MODEL_LAB_SMOKE_OUT_DIR
  || process.env.MODEL_LAB_SMOKE_OUTPUT_DIR
  || "tmp/model_lab_overview_smoke",
);
const DEBUG_PORT = Number(process.env.MODEL_LAB_SMOKE_DEBUG_PORT || 9228);

const VIEWPORTS = [
  { name: "desktop", width: 1440, height: 1100, mobile: false, deviceScaleFactor: 1 },
  { name: "mobile", width: 390, height: 844, mobile: true, deviceScaleFactor: 2 },
];

const EDGE_CANDIDATES = [
  process.env.EDGE_PATH,
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
].filter(Boolean);

function sleep(ms) {
  return new Promise((resolveSleep) => setTimeout(resolveSleep, ms));
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}: ${url}`);
  return response.json();
}

async function findBrowserExecutable() {
  const { access } = await import("node:fs/promises");
  for (const candidate of EDGE_CANDIDATES) {
    try {
      await access(candidate);
      return candidate;
    } catch (_) {
      // try next candidate
    }
  }
  throw new Error(`No Edge/Chrome executable found. Tried: ${EDGE_CANDIDATES.join(", ")}`);
}

async function waitForDebugEndpoint() {
  const versionUrl = `http://127.0.0.1:${DEBUG_PORT}/json/version`;
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    try {
      return await fetchJson(versionUrl);
    } catch (_) {
      await sleep(250);
    }
  }
  throw new Error(`Timed out waiting for Chrome DevTools endpoint on port ${DEBUG_PORT}`);
}

function connectCdp(wsUrl) {
  const ws = new WebSocket(wsUrl);
  let nextId = 1;
  const pending = new Map();
  const listeners = new Map();
  ws.onmessage = (event) => {
    const message = JSON.parse(String(event.data));
    if (message.id && pending.has(message.id)) {
      const { resolve: resolvePending, reject } = pending.get(message.id);
      pending.delete(message.id);
      if (message.error) reject(new Error(`${message.error.message || "CDP error"} (${message.error.code})`));
      else resolvePending(message.result || {});
      return;
    }
    if (message.method && listeners.has(message.method)) {
      for (const listener of listeners.get(message.method)) listener(message.params || {});
    }
  };
  const opened = new Promise((resolveOpen, rejectOpen) => {
    ws.onopen = resolveOpen;
    ws.onerror = () => rejectOpen(new Error(`CDP websocket failed: ${wsUrl}`));
  });
  return {
    async ready() {
      await opened;
    },
    send(method, params = {}) {
      const id = nextId++;
      ws.send(JSON.stringify({ id, method, params }));
      return new Promise((resolveSend, rejectSend) => pending.set(id, { resolve: resolveSend, reject: rejectSend }));
    },
    on(method, listener) {
      if (!listeners.has(method)) listeners.set(method, []);
      listeners.get(method).push(listener);
    },
    close() {
      ws.close();
    },
  };
}

async function openTarget(url) {
  const target = await fetchJson(
    `http://127.0.0.1:${DEBUG_PORT}/json/new?${encodeURIComponent(url)}`,
    { method: "PUT" },
  );
  return target.webSocketDebuggerUrl;
}

async function waitForPageReady(cdp) {
  const deadline = Date.now() + 15000;
  while (Date.now() < deadline) {
    const result = await cdp.send("Runtime.evaluate", {
      expression: `(() => {
        const text = document.body ? document.body.innerText : "";
        return {
          readyState: document.readyState,
          hasOverview: Boolean(document.querySelector(".viewer-box--overview")),
          chartCards: document.querySelectorAll("#viewer .chart-card").length,
          hasError: /Boot failed|加载失败|overview fixture 读取失败|fixture failed/i.test(text),
          stillLoading: /加载 overview fixture|加载 fixture snapshot/i.test(text),
          textLength: text.length,
        };
      })()`,
      returnByValue: true,
    });
    const value = result.result && result.result.value;
    if (
      value
      && value.readyState === "complete"
      && value.hasOverview
      && value.chartCards > 0
      && !value.stillLoading
      && value.textLength > 500
    ) return value;
    if (value && value.hasError) return value;
    await sleep(300);
  }
  throw new Error("Timed out waiting for model-lab overview fixture to render");
}

async function analyzeLayout(cdp) {
  const expression = `(() => {
    const isVisible = (el) => {
      if (el.closest("details:not([open])")) return false;
      const style = getComputedStyle(el);
      const rect = el.getBoundingClientRect();
      return style.display !== "none"
        && style.visibility !== "hidden"
        && Number(style.opacity || 1) !== 0
        && rect.width > 1
        && rect.height > 1;
    };
    const labelFor = (el) => {
      const text = (el.innerText || el.textContent || "").replace(/\\s+/g, " ").trim();
      const cls = String(el.className || "").replace(/\\s+/g, ".");
      const id = String(el.id || "");
      return {
        tag: el.tagName.toLowerCase(),
        id: id.slice(0, 80),
        className: cls.slice(0, 120),
        text: text.slice(0, 220),
      };
    };
    const ignoreOverflow = (el) => {
      if (["PRE", "CODE", "TEXTAREA", "SELECT", "OPTION"].includes(el.tagName)) return true;
      if (el.closest(".artifact-viewer-raw, .md-box, .run-table-wrap, .artifact-table-wrap, .table-wrap, .compare-table-wrap, .raw-log, .source-code")) return true;
      return false;
    };
    const hasScrollableXAncestor = (el) => {
      let node = el.parentElement;
      while (node && node !== document.body) {
        const style = getComputedStyle(node);
        const overflowX = style.overflowX;
        if ((overflowX === "auto" || overflowX === "scroll") && node.scrollWidth > node.clientWidth + 4) return true;
        node = node.parentElement;
      }
      return false;
    };
    const textOverflow = [];
    const offscreen = [];
    const blankModules = [];
    const placeholders = [];
    for (const el of Array.from(document.querySelectorAll("body *"))) {
      if (!isVisible(el)) continue;
      const rect = el.getBoundingClientRect();
      const text = (el.innerText || el.textContent || "").replace(/\\s+/g, " ").trim();
      if (text && !ignoreOverflow(el)) {
        const overflowX = el.scrollWidth - el.clientWidth;
        const overflowY = el.scrollHeight - el.clientHeight;
        if ((overflowX > 3 || overflowY > 6) && rect.width >= 24 && rect.height >= 12) {
          const style = getComputedStyle(el);
          const allowY = ["auto", "scroll"].includes(style.overflowY);
          const allowX = ["auto", "scroll"].includes(style.overflowX);
          if (!(allowX && overflowX > 0) && !(allowY && overflowY > 0)) {
            textOverflow.push({ ...labelFor(el), overflowX, overflowY, width: Math.round(rect.width), height: Math.round(rect.height) });
          }
        }
      }
      if (text && rect.right > window.innerWidth + 4 && rect.left < window.innerWidth && !hasScrollableXAncestor(el)) {
        offscreen.push({ ...labelFor(el), right: Math.round(rect.right), viewport: window.innerWidth });
      }
      if (/\\b(NaN|Infinity|undefined|\\[object Object\\])\\b/.test(text)) {
        placeholders.push(labelFor(el));
      }
    }
    for (const el of Array.from(document.querySelectorAll("#viewer .chart-card,#viewer .diag-section,#viewer .overview-primary-grid,#viewer .overview-chart-grid"))) {
      if (!isVisible(el)) continue;
      const text = (el.innerText || el.textContent || "").replace(/\\s+/g, " ").trim();
      const hasVisual = Boolean(el.querySelector("svg,canvas,img,table,.metrics-grid,.integrity-list,.coverage-event-strip,.diag-heatmap,.artifact-btn"));
      if (!text && !hasVisual) blankModules.push(labelFor(el));
      if (/加载\\.\\.\\.|Loading\\.\\.\\./i.test(text)) blankModules.push(labelFor(el));
    }
    const viewerText = (document.querySelector("#viewer")?.innerText || "").replace(/\\s+/g, " ").trim();
    return {
      title: document.title,
      url: location.href,
      viewerTitle: document.querySelector("#viewerTitle")?.textContent || "",
      bodyTextLength: (document.body && document.body.innerText || "").length,
      viewerTextLength: viewerText.length,
      viewport: { width: window.innerWidth, height: window.innerHeight },
      scrollWidth: document.documentElement.scrollWidth,
      horizontalOverflow: Math.max(0, document.documentElement.scrollWidth - window.innerWidth),
      textOverflow: textOverflow.slice(0, 30),
      offscreen: offscreen.slice(0, 30),
      blankModules: blankModules.slice(0, 30),
      placeholders: placeholders.slice(0, 30),
      overviewShellCount: document.querySelectorAll(".viewer-box--overview").length,
      primaryGridCount: document.querySelectorAll("#viewer .overview-primary-grid").length,
      chartCardCount: document.querySelectorAll("#viewer .chart-card").length,
      svgCount: document.querySelectorAll("#viewer svg").length,
      tableCount: document.querySelectorAll("#viewer table").length,
      openDetailsCount: document.querySelectorAll("details[open]").length,
    };
  })()`;
  const result = await cdp.send("Runtime.evaluate", { expression, returnByValue: true });
  return result.result.value;
}

async function captureScreenshot(cdp, outputPath) {
  const metrics = await cdp.send("Page.getLayoutMetrics");
  const contentSize = metrics.contentSize || { x: 0, y: 0, width: 1200, height: 1200 };
  const width = Math.ceil(Math.min(Math.max(contentSize.width, 1), 2400));
  const height = Math.ceil(Math.min(Math.max(contentSize.height, 1), 22000));
  const screenshot = await cdp.send("Page.captureScreenshot", {
    format: "png",
    fromSurface: true,
    captureBeyondViewport: true,
    clip: { x: 0, y: 0, width, height, scale: 1 },
  });
  await writeFile(outputPath, Buffer.from(screenshot.data, "base64"));
}

async function smokePage({ fixtureId, url, viewport }) {
  const wsUrl = await openTarget("about:blank");
  const cdp = connectCdp(wsUrl);
  await cdp.ready();
  const consoleErrors = [];
  cdp.on("Runtime.exceptionThrown", (params) => {
    consoleErrors.push({ type: "exception", message: params.exceptionDetails?.text || "Runtime exception" });
  });
  cdp.on("Log.entryAdded", (params) => {
    const entry = params.entry || {};
    if (entry.level === "error") consoleErrors.push({ type: "log", message: entry.text || "" });
  });
  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");
  await cdp.send("Log.enable");
  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width: viewport.width,
    height: viewport.height,
    deviceScaleFactor: viewport.deviceScaleFactor,
    mobile: viewport.mobile,
  });
  await cdp.send("Page.navigate", { url });
  await waitForPageReady(cdp);
  await sleep(700);
  const layout = await analyzeLayout(cdp);
  const screenshotPath = join(OUT_DIR, `${fixtureId}-${viewport.name}.png`);
  await captureScreenshot(cdp, screenshotPath);
  cdp.close();
  return { fixtureId, viewport: viewport.name, url, screenshotPath, consoleErrors, layout };
}

async function main() {
  await rm(OUT_DIR, { recursive: true, force: true });
  await mkdir(OUT_DIR, { recursive: true });

  const browserPath = await findBrowserExecutable();
  const userDataDir = join(OUT_DIR, "edge-profile");
  const browser = spawn(browserPath, [
    "--headless=new",
    `--remote-debugging-port=${DEBUG_PORT}`,
    `--user-data-dir=${userDataDir}`,
    "--disable-gpu",
    "--disable-extensions",
    "--no-first-run",
    "--no-default-browser-check",
    "--remote-allow-origins=*",
    "about:blank",
  ], { stdio: "ignore", detached: false });

  try {
    await waitForDebugEndpoint();
    const listing = await fetchJson(`${BASE_URL}/api/dev/model-lab/overview-fixtures`);
    const fixtures = Array.isArray(listing.fixtures) ? listing.fixtures : [];
    if (!fixtures.length) throw new Error("No model-lab overview dev fixtures were returned");
    const results = [];
    for (const fixture of fixtures) {
      for (const viewport of VIEWPORTS) {
        const url = `${BASE_URL}/dev/model-lab/overview-fixture?case=${encodeURIComponent(fixture.id)}`;
        results.push(await smokePage({ fixtureId: fixture.id, url, viewport }));
      }
    }
    const failures = [];
    for (const result of results) {
      const { layout } = result;
      if (result.consoleErrors.length) failures.push({ result, issue: "console errors", details: result.consoleErrors });
      if (layout.horizontalOverflow > 4) failures.push({ result, issue: "document horizontal overflow", details: layout.horizontalOverflow });
      if (layout.textOverflow.length) failures.push({ result, issue: "text overflow", details: layout.textOverflow });
      if (layout.offscreen.length) failures.push({ result, issue: "offscreen text", details: layout.offscreen });
      if (layout.blankModules.length) failures.push({ result, issue: "blank modules", details: layout.blankModules });
      if (layout.placeholders.length) failures.push({ result, issue: "placeholder tokens", details: layout.placeholders });
      if (layout.overviewShellCount < 1) failures.push({ result, issue: "missing overview shell", details: layout.overviewShellCount });
      if (layout.primaryGridCount < 1) failures.push({ result, issue: "missing overview primary grid", details: layout.primaryGridCount });
      if (layout.chartCardCount < 4) failures.push({ result, issue: "too few chart cards rendered", details: layout.chartCardCount });
      if (layout.svgCount < 2 && layout.tableCount < 1) failures.push({ result, issue: "no meaningful visuals rendered", details: layout });
    }
    const report = {
      ok: failures.length === 0,
      baseUrl: BASE_URL,
      outDir: OUT_DIR,
      checked: results.map((result) => ({
        fixtureId: result.fixtureId,
        viewport: result.viewport,
        screenshotPath: result.screenshotPath,
        consoleErrorCount: result.consoleErrors.length,
        layout: result.layout,
      })),
      failures,
    };
    await writeFile(join(OUT_DIR, "report.json"), JSON.stringify(report, null, 2));
    console.log(JSON.stringify(report, null, 2));
    if (failures.length) process.exitCode = 1;
  } finally {
    browser.kill();
  }
}

main().catch((error) => {
  console.error(error && error.stack ? error.stack : String(error));
  process.exit(1);
});
