import { fileURLToPath, pathToFileURL } from "node:url";
import path from "node:path";
import { createRequire } from "node:module";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const htmlPath = path.join(__dirname, "research_poster.html");
const pdfPath = path.join(__dirname, "research_poster_24x36.pdf");
const pngPath = path.join(__dirname, "research_poster_preview.png");

const browser = await chromium.launch();
const page = await browser.newPage({
  viewport: { width: 2304, height: 3456 },
  deviceScaleFactor: 1,
});

await page.goto(pathToFileURL(htmlPath).href, { waitUntil: "networkidle" });
await page.emulateMedia({ media: "print" });
await page.pdf({
  path: pdfPath,
  width: "24in",
  height: "36in",
  printBackground: true,
  preferCSSPageSize: true,
  margin: { top: "0in", right: "0in", bottom: "0in", left: "0in" },
});

await page.screenshot({ path: pngPath, fullPage: true });
await browser.close();

console.log(`Wrote ${pdfPath}`);
console.log(`Wrote ${pngPath}`);
