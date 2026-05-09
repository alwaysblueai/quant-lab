import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import ts from "typescript";

const root = path.resolve(fileURLToPath(new URL("..", import.meta.url)));
const sourcePath = path.join(root, "src", "dashboardSeries.ts");
const source = await readFile(sourcePath, "utf8");
const compiled = ts.transpileModule(source, {
  compilerOptions: {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.ES2022,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    strict: true,
  },
  fileName: sourcePath,
  reportDiagnostics: true,
});

const diagnostics = compiled.diagnostics ?? [];
assert.equal(
  diagnostics.length,
  0,
  diagnostics.map((diagnostic) => diagnostic.messageText).join("\n"),
);

const tmp = await mkdtemp(path.join(tmpdir(), "metrics-dashboard-series-"));
try {
  const modulePath = path.join(tmp, "dashboardSeries.mjs");
  await writeFile(modulePath, compiled.outputText, "utf8");
  const logic = await import(pathToFileURL(modulePath).href);

  assert.equal(logic.parseFiniteCell(""), null);
  assert.equal(logic.parseFiniteCell("NaN"), null);
  assert.equal(logic.parseFiniteCell("0"), 0);

  const fallbackRows = logic.parseTimeseriesRows(
    [
      { date: "2024-01-01", rank_ic: "", ic: "0.11", split_phase: "IS" },
      { date: "2024-01-02", rank_ic: "", ic: "0.22", split_phase: "OOS" },
      { date: "2024-01-03", rank_ic: "", ic: "", split_phase: "OOS" },
    ],
    ["rank_ic", "ic"],
  );
  assert.deepEqual(
    fallbackRows.map((row) => [row.date, row.value, row.sourceColumn, row.splitPhase]),
    [
      ["2024-01-01", 0.11, "ic", "IS"],
      ["2024-01-02", 0.22, "ic", "OOS"],
    ],
  );
  assert.equal(logic.primaryMetricLabel(fallbackRows), "IC");

  const splitRows = logic.parseTimeseriesRows(
    [
      { date: "2024-01-01", rank_ic: "0.10", split_phase: "IS" },
      { date: "2024-01-02", rank_ic: "0.90", split_phase: "EMBARGO" },
      { date: "2024-01-03", rank_ic: "0.20", split_phase: "OOS" },
      { date: "2024-01-04", rank_ic: "0.40", split_phase: "OOS" },
    ],
    ["rank_ic", "ic"],
  );
  assert.equal(logic.hasSplitBoundary(splitRows, null), true);
  assert.deepEqual(logic.buildSplitSeries(splitRows, null), [
    { date: "2024-01-01", rank_ic_is: 0.1 },
    { date: "2024-01-03", rank_ic_oos: 0.2 },
    { date: "2024-01-04", rank_ic_oos: 0.4 },
  ]);
  const splitMean = logic.splitMean(splitRows, null);
  assertAlmostEqual(splitMean.overall, 0.7000000000000001 / 3);
  assertAlmostEqual(splitMean.split, 0.3);

  const icRows = logic.parseTimeseriesRows(
    [
      { date: "2024-01-01", ic: "0.01", split_phase: "IS" },
      { date: "2024-01-02", ic: "0.01", split_phase: "EMBARGO" },
      { date: "2024-01-03", ic: "0.01", split_phase: "OOS" },
      { date: "2024-01-04", ic: "0.03", split_phase: "OOS" },
      { date: "2024-01-05", ic: "0.05", split_phase: "OOS" },
    ],
    ["ic"],
  );
  assert.equal(logic.splitIr(icRows, null).split, 1.5);

  assert.equal(
    logic.parseSplitStart({ oos_start: "2024-02-01" }),
    "2024-02-01",
  );
  assert.equal(
    logic.parseSplitStart("train <= 2024-01-31; test >= 2024-02-05"),
    "2024-02-05",
  );

  assert.deepEqual(
    logic.aggregateGroupMeans([
      { group: "1", group_return: "0.02" },
      { group: "1", group_return: "" },
      { group: "2", group_return: "0.04" },
      { group: "2", group_return: "0.06" },
    ]),
    [
      { group: "Q1", mean: 0.02 },
      { group: "Q2", mean: 0.05 },
    ],
  );
} finally {
  await rm(tmp, { recursive: true, force: true });
}

function assertAlmostEqual(actual, expected, epsilon = 1e-12) {
  assert.equal(typeof actual, "number");
  assert.ok(
    Math.abs(actual - expected) <= epsilon,
    `expected ${actual} to be within ${epsilon} of ${expected}`,
  );
}
