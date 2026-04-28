import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

interface ArxivPaper {
  id: string;
  title: string;
  summary: string;
  authors: string[];
  categories: string[];
  primaryCategory: string;
  published: string;
  updated: string;
  url: string;
}

const BASE_URL = "https://export.arxiv.org/api/query";
const SEARCH_QUERY =
  "(cat:cs.LG OR cat:stat.ML OR cat:cs.AI OR cat:cs.CL OR cat:cs.CV OR cat:cs.NE)";
const TOTAL_RESULTS = 500;
const PAGE_SIZE = 100;
const REQUEST_DELAY_MS = 3500;
const USER_AGENT = "vector-lens/1.0 (real arXiv seed fetch for demo dataset)";

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function decodeXml(text: string): string {
  return text
    .replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'")
    .replace(/\s+/g, " ")
    .trim();
}

function extractSingle(block: string, tag: string): string {
  const match = block.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, "i"));
  return match ? decodeXml(match[1]) : "";
}

function extractAll(block: string, tag: string): string[] {
  return Array.from(
    block.matchAll(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)<\\/${tag}>`, "gi")),
    (match) => decodeXml(match[1])
  );
}

function extractCategories(block: string): string[] {
  const categories = Array.from(
    block.matchAll(/<category[^>]*term="([^"]+)"[^>]*\/>/gi),
    (match) => decodeXml(match[1])
  );
  return Array.from(new Set(categories));
}

function extractPrimaryCategory(block: string): string {
  const match = block.match(/<arxiv:primary_category[^>]*term="([^"]+)"[^>]*\/>/i);
  return match ? decodeXml(match[1]) : "";
}

function parseFeed(xml: string): ArxivPaper[] {
  const entries = xml.match(/<entry>([\s\S]*?)<\/entry>/gi) ?? [];
  return entries
    .map((entry) => {
      const rawId = extractSingle(entry, "id");
      const id = rawId.split("/abs/").pop()?.trim() ?? rawId.trim();
      const title = extractSingle(entry, "title");
      const summary = extractSingle(entry, "summary");
      const authors = extractAll(entry, "name");
      const categories = extractCategories(entry);
      const primaryCategory = extractPrimaryCategory(entry) || categories[0] || "";
      const published = extractSingle(entry, "published");
      const updated = extractSingle(entry, "updated");

      return {
        id,
        title,
        summary,
        authors,
        categories,
        primaryCategory,
        published,
        updated,
        url: rawId,
      };
    })
    .filter((paper) => paper.id && paper.title && paper.summary && paper.authors.length > 0);
}

async function fetchPage(start: number, maxResults: number): Promise<ArxivPaper[]> {
  const params = new URLSearchParams({
    search_query: SEARCH_QUERY,
    start: String(start),
    max_results: String(maxResults),
    sortBy: "submittedDate",
    sortOrder: "descending",
  });

  const response = await fetch(`${BASE_URL}?${params.toString()}`, {
    headers: {
      "User-Agent": USER_AGENT,
      Accept: "application/atom+xml, application/xml, text/xml;q=0.9, */*;q=0.8",
    },
  });

  if (!response.ok) {
    throw new Error(`arXiv API request failed with ${response.status} ${response.statusText}`);
  }

  const xml = await response.text();
  return parseFeed(xml);
}

async function main() {
  const papers: ArxivPaper[] = [];

  for (let start = 0; start < TOTAL_RESULTS; start += PAGE_SIZE) {
    const batchSize = Math.min(PAGE_SIZE, TOTAL_RESULTS - start);
    console.log(`[fetch:arxiv] Fetching ${start}-${start + batchSize - 1}...`);
    const page = await fetchPage(start, batchSize);
    papers.push(...page);
    if (start + PAGE_SIZE < TOTAL_RESULTS) {
      await sleep(REQUEST_DELAY_MS);
    }
  }

  const deduped = Array.from(new Map(papers.map((paper) => [paper.id, paper])).values()).slice(
    0,
    TOTAL_RESULTS
  );

  const here = path.dirname(fileURLToPath(import.meta.url));
  const outDir = path.join(here, "..", "data");
  const outPath = path.join(outDir, "arxiv-ml-500.json");

  mkdirSync(outDir, { recursive: true });
  writeFileSync(outPath, `${JSON.stringify(deduped, null, 2)}\n`, "utf8");

  console.log(`[fetch:arxiv] Wrote ${deduped.length} papers to ${outPath}`);
}

main().catch((error) => {
  console.error("[fetch:arxiv] Failed:", error);
  process.exitCode = 1;
});
