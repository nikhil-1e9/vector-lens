import { MCPServer, text, widget, error } from "mcp-use/server";
import { z } from "zod";
import { readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const server = new MCPServer({
  name: "vector-lens",
  title: "Vector Lens",
  version: "1.0.0",
  description: "Visual RAG retrieval inspector for ML engineers",
  baseUrl: process.env.MCP_URL || "http://localhost:3000",
  favicon: "favicon.ico",
  icons: [{ src: "icon.svg", mimeType: "image/svg+xml", sizes: ["512x512"] }],
});

// --- Mock arXiv ML paper abstracts dataset ---

const EMBEDDING_DIM = 128;
const SEED = 42;

interface Chunk {
  id: string;
  text: string;
  embedding: number[];
  source: string;
  year: number;
  authors: string;
  topics: string[];
  x: number;
  y: number;
}

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

// Simple seeded PRNG
function mulberry32(seed: number) {
  let s = seed;
  return () => {
    s |= 0;
    s = (s + 0x6d2b79f5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Hash a string to an integer
function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

function tokenize(txt: string): string[] {
  return txt
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter(Boolean);
}

function normalize(vec: number[]): number[] {
  let norm = 0;
  for (let i = 0; i < vec.length; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  return vec.map((v) => v / norm);
}

function deterministicVector(key: string, dim = EMBEDDING_DIM): number[] {
  const localRng = mulberry32(hashStr(`${SEED}:${key}`));
  const vec = new Array(dim).fill(0).map(() => localRng() * 2 - 1);
  return normalize(vec);
}

// Generate a deterministic embedding from text via bag-of-words hashing
function embedTextResidual(txt: string): number[] {
  const vec = new Float64Array(EMBEDDING_DIM);
  const words = tokenize(txt);
  for (const w of words) {
    const h = hashStr(w);
    const dim1 = h % EMBEDDING_DIM;
    const dim2 = (h * 31) % EMBEDDING_DIM;
    const dim3 = (h * 97) % EMBEDDING_DIM;
    vec[dim1] += 1.0;
    vec[dim2] += 0.5;
    vec[dim3] += 0.25;
  }
  // Normalize to unit vector
  const result: number[] = [];
  for (let i = 0; i < EMBEDDING_DIM; i++) result.push(vec[i]);
  return normalize(result);
}

// Cosine similarity between two unit vectors
function cosineSim(a: number[], b: number[]): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

// --- Generate mock dataset ---

const topics = [
  "transformer", "attention mechanism", "self-supervised learning", "contrastive learning",
  "generative adversarial network", "diffusion model", "reinforcement learning", "policy gradient",
  "natural language processing", "language model", "computer vision", "object detection",
  "image segmentation", "graph neural network", "federated learning", "meta-learning",
  "few-shot learning", "knowledge distillation", "neural architecture search", "pruning",
  "quantization", "multi-modal learning", "vision transformer", "BERT", "GPT",
  "variational autoencoder", "normalizing flow", "optimal transport", "causal inference",
  "bayesian optimization", "active learning", "curriculum learning", "data augmentation",
  "domain adaptation", "transfer learning", "representation learning", "embedding",
  "recommender system", "time series forecasting", "speech recognition",
];

const topicAliases: Record<string, string[]> = {
  transformer: ["transformer", "transformers"],
  "attention mechanism": ["attention", "attention mechanism", "self-attention"],
  "self-supervised learning": ["self-supervised learning", "ssl"],
  "contrastive learning": ["contrastive learning", "contrastive"],
  "generative adversarial network": ["generative adversarial network", "gan", "gans"],
  "diffusion model": ["diffusion", "diffusion model", "diffusion models"],
  "reinforcement learning": ["reinforcement learning", "rl"],
  "policy gradient": ["policy gradient", "policy gradients"],
  "natural language processing": ["natural language processing", "nlp"],
  "language model": ["language model", "llm", "language models"],
  "computer vision": ["computer vision", "vision"],
  "object detection": ["object detection", "detector"],
  "image segmentation": ["image segmentation", "segmentation"],
  "graph neural network": ["graph neural network", "graph neural networks", "gnn", "gnns"],
  "federated learning": ["federated learning"],
  "meta-learning": ["meta-learning", "metalearning"],
  "few-shot learning": ["few-shot", "few-shot learning"],
  "knowledge distillation": ["knowledge distillation", "distillation"],
  "neural architecture search": ["neural architecture search", "nas"],
  pruning: ["pruning", "sparsity"],
  quantization: ["quantization", "quantized", "quantisation"],
  "multi-modal learning": ["multimodal", "multi-modal", "multi-modal learning"],
  "vision transformer": ["vision transformer", "vit"],
  BERT: ["bert"],
  GPT: ["gpt"],
  "variational autoencoder": ["variational autoencoder", "vae"],
  "normalizing flow": ["normalizing flow", "flows"],
  "optimal transport": ["optimal transport"],
  "causal inference": ["causal inference", "causality"],
  "bayesian optimization": ["bayesian optimization"],
  "active learning": ["active learning"],
  "curriculum learning": ["curriculum learning"],
  "data augmentation": ["data augmentation", "augmentation"],
  "domain adaptation": ["domain adaptation"],
  "transfer learning": ["transfer learning"],
  "representation learning": ["representation learning", "embeddings"],
  embedding: ["embedding", "embeddings", "vector search", "semantic search"],
  "recommender system": ["recommender system", "recommendation"],
  "time series forecasting": ["time series", "forecasting"],
  "speech recognition": ["speech recognition", "asr"],
};

const categoryTopics: Record<string, string[]> = {
  "cs.LG": ["representation learning", "transfer learning", "active learning"],
  "stat.ML": ["bayesian optimization", "causal inference", "representation learning"],
  "cs.AI": ["language model", "reinforcement learning", "graph neural network"],
  "cs.CL": ["natural language processing", "language model", "BERT", "GPT"],
  "cs.CV": ["computer vision", "object detection", "image segmentation", "vision transformer"],
  "cs.NE": ["meta-learning", "few-shot learning", "knowledge distillation"],
  "cs.IR": ["embedding", "recommender system"],
  "cs.SD": ["speech recognition", "time series forecasting"],
  "eess.AS": ["speech recognition"],
};

const topicBasis = new Map(
  topics.map((topic) => [topic, deterministicVector(`topic:${topic}`)])
);

const topicAnchors = new Map(
  topics.map((topic, i) => {
    const angle = (i / topics.length) * Math.PI * 2;
    const radius = 4.2 + (i % 4) * 0.35;
    return [topic, { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius }];
  })
);

const projectionBasisX = deterministicVector("projection:x");
const projectionBasisY = deterministicVector("projection:y");

function detectTopics(text: string): string[] {
  const normalized = text.toLowerCase();
  const matched = topics.filter((topic) =>
    (topicAliases[topic] ?? [topic.toLowerCase()]).some((alias) => normalized.includes(alias))
  );
  return matched;
}

function inferTopicsFromPaper(paper: ArxivPaper): string[] {
  const textTopics = detectTopics(`${paper.title} ${paper.summary}`);
  const categoryHints = paper.categories.flatMap((category) => categoryTopics[category] ?? []);
  return Array.from(new Set([...textTopics, ...categoryHints])).slice(0, 6);
}

function embedSemanticText(text: string, topicHints: string[], source?: string): number[] {
  const vec = new Array(EMBEDDING_DIM).fill(0);
  const residual = embedTextResidual(text);
  const titleResidual = source ? embedTextResidual(source) : null;

  for (let i = 0; i < EMBEDDING_DIM; i++) {
    vec[i] += residual[i] * 0.55;
    if (titleResidual) vec[i] += titleResidual[i] * 0.25;
  }

  topicHints.forEach((topic, idx) => {
    const basis = topicBasis.get(topic);
    if (!basis) return;
    const weight = idx === 0 ? 1.25 : 0.9;
    for (let i = 0; i < EMBEDDING_DIM; i++) vec[i] += basis[i] * weight;
  });

  return normalize(vec);
}

function projectPoint(
  embedding: number[],
  topicHints: string[],
  idSeed: string
): { x: number; y: number } {
  let x = 0;
  let y = 0;

  if (topicHints.length) {
    for (const topic of topicHints) {
      const anchor = topicAnchors.get(topic);
      if (!anchor) continue;
      x += anchor.x;
      y += anchor.y;
    }
    x /= topicHints.length;
    y /= topicHints.length;
  }

  let residualX = 0;
  let residualY = 0;
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    residualX += embedding[i] * projectionBasisX[i];
    residualY += embedding[i] * projectionBasisY[i];
  }

  const noiseRng = mulberry32(hashStr(idSeed));
  x += residualX * 0.9 + (noiseRng() - 0.5) * 0.45;
  y += residualY * 0.9 + (noiseRng() - 0.5) * 0.45;

  return { x, y };
}

function loadArxivPapers(): ArxivPaper[] {
  const here = path.dirname(fileURLToPath(import.meta.url));
  const datasetPath = path.join(here, "data", "arxiv-ml-500.json");

  try {
    const raw = readFileSync(datasetPath, "utf8");
    const parsed = JSON.parse(raw) as ArxivPaper[];
    if (!Array.isArray(parsed) || parsed.length === 0) {
      throw new Error("Dataset is empty");
    }
    return parsed;
  } catch (cause) {
    throw new Error(
      `Missing real arXiv dataset at ${datasetPath}. Run \`npm run fetch:arxiv\` to generate it.`,
      { cause }
    );
  }
}

function generateChunks(): Chunk[] {
  return loadArxivPapers().map((paper) => {
    const chunkTopics = inferTopicsFromPaper(paper);
    return {
      id: paper.id,
      text: paper.summary,
      embedding: embedSemanticText(paper.summary, chunkTopics, paper.title),
      source: paper.title,
      year: new Date(paper.published).getUTCFullYear(),
      authors: paper.authors.join(", "),
      topics: chunkTopics,
      x: 0,
      y: 0,
    };
  });
}

function computeProjection(chunks: Chunk[]): void {
  for (const chunk of chunks) {
    const point = projectPoint(chunk.embedding, chunk.topics, chunk.id);
    chunk.x = point.x;
    chunk.y = point.y;
  }
}

// --- Initialize dataset ---
console.log("[Vector Lens] Loading real arXiv ML abstract chunks...");
const chunks = generateChunks();
computeProjection(chunks);
console.log("[Vector Lens] Dataset ready. UMAP projection computed.");

const chunkTokens = new Map<string, Set<string>>();
const tokenDocumentFrequency = new Map<string, number>();

for (const chunk of chunks) {
  const tokens = new Set(tokenize(`${chunk.source} ${chunk.text}`));
  chunkTokens.set(chunk.id, tokens);
  for (const token of tokens) {
    tokenDocumentFrequency.set(token, (tokenDocumentFrequency.get(token) ?? 0) + 1);
  }
}

function lexicalScore(query: string, chunk: Chunk): number {
  const qTokens = Array.from(new Set(tokenize(query)));
  if (!qTokens.length) return 0;

  const docTokens = chunkTokens.get(chunk.id) ?? new Set<string>();
  let matchedWeight = 0;
  let totalWeight = 0;

  for (const token of qTokens) {
    const df = tokenDocumentFrequency.get(token) ?? 0;
    const idf = Math.log(1 + chunks.length / (1 + df));
    totalWeight += idf;
    if (docTokens.has(token)) matchedWeight += idf;
  }

  const phrase = query.trim().toLowerCase();
  const phraseHit =
    phrase.length > 3 && `${chunk.source} ${chunk.text}`.toLowerCase().includes(phrase)
      ? 0.18
      : 0;

  return Math.min(1, matchedWeight / Math.max(totalWeight, 1) + phraseHit);
}

function topicOverlapScore(queryTopics: string[], chunk: Chunk): number {
  if (!queryTopics.length) return 0;
  let overlap = 0;
  for (const topic of queryTopics) {
    if (chunk.topics.includes(topic)) overlap++;
  }
  return overlap / queryTopics.length;
}

// --- Search tool ---
server.tool(
  {
    name: "search",
    description:
      "Search the arXiv ML abstracts index using semantic similarity. Returns top-k results with similarity scores and embedding space data for visualization.",
    schema: z.object({
      query: z.string().describe("Natural language search query for ML paper abstracts"),
      k: z.number().min(1).max(50).optional().describe("Number of top results to return (default: 5)"),
    }),
    widget: {
      name: "vector-lens",
      invoking: "Embedding query and searching index...",
      invoked: "Search complete",
    },
  },
  async ({ query, k = 5 }) => {
    const startTime = performance.now();

    const queryTopics = detectTopics(query);
    const queryEmbedding = embedSemanticText(query, queryTopics, query);
    const queryPoint = projectPoint(queryEmbedding, queryTopics, `query:${query}`);

    const scored = chunks.map((c) => ({
      id: c.id,
      text: c.text,
      source: c.source,
      year: c.year,
      authors: c.authors,
      topics: c.topics,
      denseScore: Math.max(0, cosineSim(queryEmbedding, c.embedding)),
      lexicalScore: lexicalScore(query, c),
      topicOverlap: topicOverlapScore(queryTopics, c),
      x: c.x,
      y: c.y,
    })).map((result) => {
      const similarity = Math.min(
        1,
        result.denseScore * 0.6 +
          result.lexicalScore * 0.28 +
          result.topicOverlap * 0.12
      );
      return {
        ...result,
        similarity,
      };
    });

    // Sort by similarity descending
    scored.sort((a, b) => b.similarity - a.similarity);

    const topK = scored.slice(0, k);
    const topKIds = new Set(topK.map((r) => r.id));

    const latencyMs = Math.round(performance.now() - startTime);

    // Pre-bin similarities for histogram (25 bins)
    const NUM_BINS = 25;
    const histBins: number[] = new Array(NUM_BINS).fill(0);
    for (const s of scored) {
      const idx = Math.min(NUM_BINS - 1, Math.floor(s.similarity * NUM_BINS));
      histBins[idx]++;
    }

    // Embedding space points (downsample to ~150 for widget perf)
    const step = Math.max(1, Math.floor(scored.length / 150));
    const spacePoints = scored
      .filter((_, i) => i < k || i % step === 0)
      .map((s) => ({
        id: s.id,
        x: s.x,
        y: s.y,
        sim: parseFloat(s.similarity.toFixed(3)),
        isTopK: topKIds.has(s.id),
        source: s.source,
        year: s.year,
        topics: s.topics,
        rank: topKIds.has(s.id) ? topK.findIndex((item) => item.id === s.id) + 1 : null,
        preview: s.text.slice(0, 80),
      }));

    const props = {
      query,
      k,
      results: topK.map((r) => ({
        id: r.id,
        text: r.text,
        source: r.source,
        year: r.year,
        authors: r.authors,
        topics: r.topics,
        similarity: parseFloat(r.similarity.toFixed(4)),
        denseScore: parseFloat(r.denseScore.toFixed(4)),
        lexicalScore: parseFloat(r.lexicalScore.toFixed(4)),
        topicOverlap: parseFloat(r.topicOverlap.toFixed(4)),
      })),
      spacePoints,
      queryPoint: { x: queryPoint.x, y: queryPoint.y, topics: queryTopics },
      stats: {
        indexSize: chunks.length,
        embeddingDim: EMBEDDING_DIM,
        distanceMetric: "hybrid" as const,
        latencyMs,
        top1Sim: parseFloat(topK[0]?.similarity.toFixed(4) ?? "0"),
        topKSimGap: parseFloat(
          ((topK[0]?.similarity ?? 0) - (topK[topK.length - 1]?.similarity ?? 0)).toFixed(4)
        ),
        scoringMode: "0.60 dense + 0.28 lexical + 0.12 topic prior",
        matchedTopics: queryTopics,
      },
      histBins,
      modelName: "Hybrid semantic + lexical ranker",
      datasetName: "Real arXiv ML Abstracts",
    };

    const summaryLines = topK
      .slice(0, 3)
      .map((r, i) => `${i + 1}. [${r.similarity.toFixed(3)}] ${r.source}`)
      .join("\n");

    return widget({
      props,
      output: text(
        `Search: "${query}" | Top ${k} of ${chunks.length} chunks | Latency: ${latencyMs}ms\n\n${summaryLines}`
      ),
    });
  }
);

const PORT = process.env.PORT ? parseInt(process.env.PORT) : 3000;
console.log(`Server running on port ${PORT}`);
server.listen(PORT);
