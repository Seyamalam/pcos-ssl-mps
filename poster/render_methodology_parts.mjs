import { fileURLToPath } from "node:url";
import fs from "node:fs/promises";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const diagrams = [
  ["methodology_part1_data.mmd", "methodology_part1_data.png"],
  ["methodology_part2_training.mmd", "methodology_part2_training.png"],
  ["methodology_part3_reliability.mmd", "methodology_part3_reliability.png"],
];

const browser = await chromium.launch();

for (const [sourceName, outputName] of diagrams) {
  const source = await fs.readFile(path.join(__dirname, sourceName), "utf8");
  const page = await browser.newPage({
    viewport: { width: 920, height: 980 },
    deviceScaleFactor: 2,
  });

  await page.setContent(
    `<!doctype html>
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          html, body {
            margin: 0;
            width: 920px;
            height: 980px;
            background: #ffffff;
            font-family: Inter, Arial, sans-serif;
          }
          .wrap {
            width: 920px;
            height: 980px;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 22px;
            box-sizing: border-box;
          }
          .mermaid { width: 100%; }
          .mermaid svg {
            width: 100% !important;
            height: auto !important;
            max-height: 930px;
          }
          .nodeLabel {
            line-height: 1.18 !important;
            font-weight: 800 !important;
          }
          .edgeLabel, .label {
            font-family: Inter, Arial, sans-serif !important;
          }
        </style>
      </head>
      <body>
        <div class="wrap">
          <pre class="mermaid">${source.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;")}</pre>
        </div>
        <script type="module">
          import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
          mermaid.initialize({ startOnLoad: true, securityLevel: "loose" });
          await mermaid.run();
          window.__MERMAID_DONE__ = true;
        </script>
      </body>
    </html>`,
    { waitUntil: "networkidle" },
  );

  await page.waitForFunction(() => window.__MERMAID_DONE__ === true, null, { timeout: 30000 });
  await page.locator(".mermaid svg").screenshot({
    path: path.join(__dirname, outputName),
    omitBackground: false,
  });
  await page.close();
  console.log(`Wrote ${path.join(__dirname, outputName)}`);
}

await browser.close();
