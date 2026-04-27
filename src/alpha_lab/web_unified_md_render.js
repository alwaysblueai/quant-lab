function mdRender(text) {
  const SENTINEL = "@@BLK";
  const MATH_SENT = "@@MTH";
  function esc(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
  function isTableRow(line) {
    const s = String(line || "").trim();
    return s.includes("|") && /^\|?.+\|.+\|?$/.test(s);
  }
  function isTableSeparator(line) {
    const s = String(line || "").trim();
    if (!s.includes("|")) return false;
    const body = s.replace(/^\|/, "").replace(/\|$/, "");
    const cells = body.split("|").map((cell) => cell.trim());
    return cells.length >= 2 && cells.every((cell) => /^:?-{3,}:?$/.test(cell));
  }
  function parseTableCells(line) {
    return String(line || "")
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim());
  }
  function inline(s) {
    s = esc(s);
    // Wikilinks — replace with placeholders first to protect from later regexes
    var wikiHolds = [];
    s = s.replace(/\[\[(.+?)(?:\|(.+?))?\]\]/g, function(_, page, alias) {
      var idx = wikiHolds.length;
      var label = (alias || page).trim();
      var cardPath = page.trim() + (page.trim().endsWith(".md") ? "" : ".md");
      wikiHolds.push('<span class="wikilink" data-action="selectCard" data-card-path="' +
        cardPath.replace(/"/g, "&quot;") + '" style="cursor:pointer">' + label + '</span>');
      return "@@WLK" + idx + "@@";
    });
    s = s.replace(/\*\*\*(.+?)\*\*\*/g, "<strong><em>$1</em></strong>");
    s = s.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/\*(.+?)\*/g, "<em>$1</em>");
    // Treat underscores as emphasis only at token boundaries so snake_case
    // identifiers such as asym_vol_reversal_v1 remain intact.
    s = s.replace(
      /(^|[^0-9A-Za-z])_([^_\n]+?)_(?=[^0-9A-Za-z]|$)/g,
      function(_, prefix, inner) {
        return prefix + "<em>" + inner + "</em>";
      },
    );
    s = s.replace(/~~(.+?)~~/g, "<del>$1</del>");
    s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
    s = s.replace(
      /\[(.+?)\]\(([^)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener">$1</a>',
    );
    s = s.replace(/\n/g, "<br>");
    // Restore wikilinks after all inline regexes
    s = s.replace(/@@WLK(\d+)@@/g, function(_, i) { return wikiHolds[+i]; });
    return s;
  }
  // 1. Protect fenced code blocks
  const blocks = [];
  text = text.replace(/```[^\n]*\n([\s\S]*?)```/g, function(_, code) {
    const idx = blocks.length;
    blocks.push(
      '<pre class="code-block"><code>' +
      esc(code.replace(/\n$/, "")) +
      "</code></pre>"
    );
    return SENTINEL + idx + "@@";
  });
  // 2. Protect math blocks so inline() doesn't mangle them
  const mathBlocks = [];
  // Display math $$...$$ (must come before inline $)
  text = text.replace(/\$\$([\s\S]+?)\$\$/g, function(m) {
    const idx = mathBlocks.length; mathBlocks.push(m); return MATH_SENT + idx + "@@";
  });
  // Inline math $...$ (single line, non-empty)
  text = text.replace(/\$([^\n$]+?)\$/g, function(m) {
    const idx = mathBlocks.length; mathBlocks.push(m); return MATH_SENT + idx + "@@";
  });
  // 3. Strip YAML frontmatter
  text = text.replace(/^---\n[\s\S]*?\n---\n?/, "");
  const lines = text.split("\n");
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const raw = lines[i];
    if (raw.indexOf(SENTINEL) !== -1) {
      const m = raw.match(/@@BLK(\d+)@@/);
      if (m) { out.push(blocks[+m[1]]); i++; continue; }
    }
    if (raw.indexOf(MATH_SENT) !== -1) {
      // Math placeholder line — pass through as-is; restored to LaTeX below
      out.push(raw); i++; continue;
    }
    if (
      i + 1 < lines.length
      && isTableRow(raw)
      && isTableSeparator(lines[i + 1])
    ) {
      const header = parseTableCells(raw);
      const rows = [];
      i += 2;
      while (i < lines.length && isTableRow(lines[i]) && !isTableSeparator(lines[i])) {
        rows.push(parseTableCells(lines[i]));
        i++;
      }
      out.push(
        '<div class="artifact-table-wrap"><table><thead><tr>'
        + header.map((cell) => "<th>" + inline(cell) + "</th>").join("")
        + "</tr></thead><tbody>"
        + rows
          .map(
            (row) =>
              "<tr>"
              + header.map((_, idx) => "<td>" + inline(row[idx] || "") + "</td>").join("")
              + "</tr>"
          )
          .join("")
        + "</tbody></table></div>"
      );
      continue;
    }
    const hm = raw.match(/^(#{1,4}) +(.*)/);
    if (hm) {
      const lv = hm[1].length;
      out.push("<h" + lv + ">" + inline(hm[2]) + "</h" + lv + ">");
      i++;
      continue;
    }
    if (/^-{3,}\s*$/.test(raw)) { out.push("<hr>"); i++; continue; }
    if (raw.startsWith("> ")) {
      const bq = [];
      while (i < lines.length && lines[i].startsWith("> ")) { bq.push(lines[i].slice(2)); i++; }
      out.push("<blockquote>" + inline(bq.join("\n")) + "</blockquote>");
      continue;
    }
    if (/^[*+-] /.test(raw)) {
      const items = [];
      while (i < lines.length && /^[*+-] /.test(lines[i])) {
        items.push("<li>" + inline(lines[i].replace(/^[*+-] /, "")) + "</li>"); i++;
      }
      out.push("<ul>" + items.join("") + "</ul>"); continue;
    }
    if (/^\d+[.)]\s/.test(raw)) {
      const items = [];
      while (i < lines.length && /^\d+[.)]\s/.test(lines[i])) {
        items.push("<li>" + inline(lines[i].replace(/^\d+[.)]\s/, "")) + "</li>"); i++;
      }
      out.push("<ol>" + items.join("") + "</ol>"); continue;
    }
    if (!raw.trim()) { i++; continue; }
    const para = [];
    while (i < lines.length && lines[i].trim() &&
           !/^(#{1,4} |-{3,}|[*+-] |\d+[.)]\s|> |@@BLK|@@MTH)/.test(lines[i])) {
      para.push(lines[i]); i++;
    }
    if (para.length) out.push("<p>" + inline(para.join("\n")) + "</p>");
  }
  // 4. Restore math (leave raw LaTeX — MathJax will process it in the DOM)
  let result = out.join("\n");
  result = result.replace(/@@MTH(\d+)@@/g, function(_, idx) {
    var m = mathBlocks[+idx];
    if (m.substring(0, 2) === "$$") return '<div class="math-display">' + m + '</div>';
    return m;
  });
  return result;
}
