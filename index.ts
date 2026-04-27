import { MCPServer, text, widget, error } from "mcp-use/server";
import { z } from "zod";

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
  x: number;
  y: number;
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

const rng = mulberry32(SEED);

// Hash a string to an integer
function hashStr(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

// Generate a deterministic embedding from text via bag-of-words hashing
function embedText(txt: string): number[] {
  const vec = new Float64Array(EMBEDDING_DIM);
  const words = txt.toLowerCase().replace(/[^a-z0-9\s]/g, "").split(/\s+/).filter(Boolean);
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
  let norm = 0;
  for (let i = 0; i < EMBEDDING_DIM; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm) || 1;
  const result: number[] = [];
  for (let i = 0; i < EMBEDDING_DIM; i++) result.push(vec[i] / norm);
  return result;
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

const methods = [
  "We propose a novel approach that", "This paper introduces a new method which",
  "We present an efficient framework that", "Our work demonstrates that",
  "In this paper we develop a technique that", "We investigate how",
  "This study explores the use of", "We design a scalable system that",
  "Our approach leverages", "We introduce a principled method for",
];

const results = [
  "achieves state-of-the-art results on multiple benchmarks",
  "outperforms existing baselines by a significant margin",
  "reduces computational cost while maintaining accuracy",
  "demonstrates strong generalization to unseen domains",
  "shows consistent improvements across diverse tasks",
  "yields competitive performance with fewer parameters",
  "scales efficiently to large datasets",
  "provides theoretical guarantees on convergence",
  "enables real-time inference on edge devices",
  "improves sample efficiency by an order of magnitude",
];

const firstNames = [
  "Wei", "Yann", "Yoshua", "Geoffrey", "Ilya", "Kaiming", "Ashish", "Dario",
  "Sergey", "Alex", "Ian", "Alec", "Oriol", "Jian", "Ross", "Andrej",
  "Pieter", "Chelsea", "Sara", "Timnit", "Percy", "Christopher", "Jason",
  "Jacob", "Samy", "Hugo", "Thomas", "Richard", "David", "Michael",
];

const lastNames = [
  "Zhang", "LeCun", "Bengio", "Hinton", "Sutskever", "He", "Vaswani", "Amodei",
  "Levine", "Krizhevsky", "Goodfellow", "Radford", "Vinyals", "Sun", "Girshick",
  "Karpathy", "Abbeel", "Finn", "Hooker", "Gebru", "Liang", "Manning", "Wei",
  "Devlin", "Benoliel", "Larochelle", "Wolf", "Socher", "Silver", "Jordan",
];

function generateChunks(): Chunk[] {
  const chunks: Chunk[] = [];
  const localRng = mulberry32(123);

  for (let i = 0; i < 500; i++) {
    const topicIdx1 = Math.floor(localRng() * topics.length);
    const topicIdx2 = Math.floor(localRng() * topics.length);
    const methodIdx = Math.floor(localRng() * methods.length);
    const resultIdx = Math.floor(localRng() * results.length);

    const topic1 = topics[topicIdx1];
    const topic2 = topics[topicIdx2];
    const method = methods[methodIdx];
    const result = results[resultIdx];

    const year = 2018 + Math.floor(localRng() * 7);
    const numAuthors = 2 + Math.floor(localRng() * 3);
    const authorList: string[] = [];
    for (let a = 0; a < numAuthors; a++) {
      const fn = firstNames[Math.floor(localRng() * firstNames.length)];
      const ln = lastNames[Math.floor(localRng() * lastNames.length)];
      authorList.push(`${fn} ${ln}`);
    }

    const extraDetail = [
      `We evaluate on ${Math.floor(localRng() * 5) + 3} benchmark datasets.`,
      `Our model uses ${Math.floor(localRng() * 10 + 2)} layers with ${Math.floor(localRng() * 512 + 64)} hidden dimensions.`,
      `Experiments show a ${(localRng() * 15 + 1).toFixed(1)}% improvement over the previous best.`,
      `The training requires ${Math.floor(localRng() * 8 + 1)} GPU-hours on A100.`,
      `We release code and pretrained weights for reproducibility.`,
    ][Math.floor(localRng() * 5)];

    const abstractText = `${method} combines ${topic1} with ${topic2} for improved performance in deep learning applications. Our approach ${result}. ${extraDetail} We further analyze ablation studies demonstrating the contribution of each component. The proposed architecture introduces a novel ${topic1}-based module that integrates seamlessly with existing ${topic2} pipelines.`;

    const paperTitle = `${topic1.split(" ").map(w => w[0].toUpperCase() + w.slice(1)).join(" ")} Meets ${topic2.split(" ").map(w => w[0].toUpperCase() + w.slice(1)).join(" ")}: A Unified Framework`;

    chunks.push({
      id: `arxiv-${2000 + i}`,
      text: abstractText,
      embedding: embedText(abstractText),
      source: paperTitle,
      year,
      authors: authorList.join(", "),
      x: 0,
      y: 0,
    });
  }

  return chunks;
}

// Pre-compute UMAP-like 2D projection using random projection + topic clustering
function computeProjection(chunks: Chunk[]): void {
  // Two fixed random projection vectors
  const proj1: number[] = [];
  const proj2: number[] = [];
  const projRng = mulberry32(999);
  for (let i = 0; i < EMBEDDING_DIM; i++) {
    proj1.push(projRng() * 2 - 1);
    proj2.push(projRng() * 2 - 1);
  }

  for (const chunk of chunks) {
    let x = 0, y = 0;
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      x += chunk.embedding[i] * proj1[i];
      y += chunk.embedding[i] * proj2[i];
    }
    // Add some noise for spread
    const noiseRng = mulberry32(hashStr(chunk.id));
    x += (noiseRng() - 0.5) * 0.5;
    y += (noiseRng() - 0.5) * 0.5;
    chunk.x = x;
    chunk.y = y;
  }
}

// --- Initialize dataset ---
console.log("[Vector Lens] Generating 500 arXiv ML abstract chunks...");
const chunks = generateChunks();
computeProjection(chunks);
console.log("[Vector Lens] Dataset ready. UMAP projection computed.");

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

    // Embed the query
    const queryEmbedding = embedText(query);

    // Compute query 2D projection
    const projRng = mulberry32(999);
    const proj1: number[] = [];
    const proj2: number[] = [];
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      proj1.push(projRng() * 2 - 1);
      proj2.push(projRng() * 2 - 1);
    }
    let qx = 0, qy = 0;
    for (let i = 0; i < EMBEDDING_DIM; i++) {
      qx += queryEmbedding[i] * proj1[i];
      qy += queryEmbedding[i] * proj2[i];
    }

    // Score all chunks
    const scored = chunks.map((c) => ({
      id: c.id,
      text: c.text,
      source: c.source,
      year: c.year,
      authors: c.authors,
      similarity: Math.max(0, cosineSim(queryEmbedding, c.embedding)),
      x: c.x,
      y: c.y,
    }));

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
        similarity: parseFloat(r.similarity.toFixed(4)),
      })),
      spacePoints,
      queryPoint: { x: qx, y: qy },
      stats: {
        indexSize: chunks.length,
        embeddingDim: EMBEDDING_DIM,
        distanceMetric: "cosine" as const,
        latencyMs,
        top1Sim: parseFloat(topK[0]?.similarity.toFixed(4) ?? "0"),
        topKSimGap: parseFloat(
          ((topK[0]?.similarity ?? 0) - (topK[topK.length - 1]?.similarity ?? 0)).toFixed(4)
        ),
      },
      histBins,
      modelName: "all-MiniLM-L6-v2",
      datasetName: "arXiv ML Abstracts",
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
