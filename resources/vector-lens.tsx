import { useState, useMemo, useRef, useCallback } from "react";
import {
  McpUseProvider,
  useWidget,
  useWidgetTheme,
  useCallTool,
  type WidgetMetadata,
} from "mcp-use/react";
import { z } from "zod";

const propsSchema = z.object({
  query: z.string(),
  k: z.number(),
  results: z.array(
    z.object({
      id: z.string(),
      text: z.string(),
      source: z.string(),
      year: z.number(),
      authors: z.string(),
      similarity: z.number(),
    })
  ),
  spacePoints: z.array(
    z.object({
      id: z.string(),
      x: z.number(),
      y: z.number(),
      sim: z.number(),
      isTopK: z.boolean(),
      preview: z.string(),
    })
  ),
  queryPoint: z.object({ x: z.number(), y: z.number() }),
  stats: z.object({
    indexSize: z.number(),
    embeddingDim: z.number(),
    distanceMetric: z.string(),
    latencyMs: z.number(),
    top1Sim: z.number(),
    topKSimGap: z.number(),
  }),
  histBins: z.array(z.number()),
  modelName: z.string(),
  datasetName: z.string(),
});

export const widgetMetadata: WidgetMetadata = {
  description:
    "Visual RAG retrieval inspector showing search results, embedding space, and statistics",
  props: propsSchema,
  exposeAsTool: false,
};

type Props = z.infer<typeof propsSchema>;

type Tab = "results" | "embedding" | "stats";

// --- Color helpers ---
function simColor(sim: number): string {
  if (sim >= 0.7) return "#22c55e";
  if (sim >= 0.4) return "#eab308";
  return "#ef4444";
}

function simBgColor(sim: number): string {
  if (sim >= 0.7) return "rgba(34,197,94,0.12)";
  if (sim >= 0.4) return "rgba(234,179,8,0.12)";
  return "rgba(239,68,68,0.12)";
}

// --- Highlight matching tokens ---
function highlightText(
  txt: string,
  query: string,
  maxLen: number,
  expanded: boolean,
  accentColor: string
): React.ReactNode[] {
  const display = expanded ? txt : txt.slice(0, maxLen) + (txt.length > maxLen ? "..." : "");
  const queryWords = new Set(
    query
      .toLowerCase()
      .split(/\s+/)
      .filter((w) => w.length > 2)
  );
  const parts = display.split(/(\s+)/);
  return parts.map((part, i) => {
    const clean = part.toLowerCase().replace(/[^a-z0-9]/g, "");
    if (queryWords.has(clean)) {
      return (
        <span
          key={i}
          style={{
            backgroundColor: accentColor + "33",
            borderBottom: `2px solid ${accentColor}`,
            borderRadius: 2,
            padding: "0 2px",
          }}
        >
          {part}
        </span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

// --- Scatter Plot (SVG) ---
function ScatterPlot({
  points,
  queryPoint,
  colors,
}: {
  points: Props["spacePoints"];
  queryPoint: Props["queryPoint"];
  colors: ReturnType<typeof useColors>;
}) {
  const svgRef = useRef<SVGSVGElement>(null);
  const [tooltip, setTooltip] = useState<{
    x: number;
    y: number;
    text: string;
  } | null>(null);
  const [viewBox, setViewBox] = useState({ x: 0, y: 0, w: 0, h: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const panStart = useRef({ x: 0, y: 0, vx: 0, vy: 0 });

  // Compute bounds
  const bounds = useMemo(() => {
    const allX = [...points.map((p) => p.x), queryPoint.x];
    const allY = [...points.map((p) => p.y), queryPoint.y];
    const minX = Math.min(...allX);
    const maxX = Math.max(...allX);
    const minY = Math.min(...allY);
    const maxY = Math.max(...allY);
    const padX = (maxX - minX) * 0.1 || 1;
    const padY = (maxY - minY) * 0.1 || 1;
    return {
      x: minX - padX,
      y: minY - padY,
      w: maxX - minX + padX * 2,
      h: maxY - minY + padY * 2,
    };
  }, [points, queryPoint]);

  // Initialize viewBox
  useMemo(() => {
    if (viewBox.w === 0) setViewBox(bounds);
  }, [bounds, viewBox.w]);

  const vb = viewBox.w > 0 ? viewBox : bounds;
  const WIDTH = 600;
  const HEIGHT = 400;

  const toSvgX = useCallback(
    (px: number) => ((px - vb.x) / vb.w) * WIDTH,
    [vb]
  );
  const toSvgY = useCallback(
    (py: number) => ((py - vb.y) / vb.h) * HEIGHT,
    [vb]
  );

  const handleWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1.15 : 0.87;
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return;
      const mx = ((e.clientX - rect.left) / rect.width) * vb.w + vb.x;
      const my = ((e.clientY - rect.top) / rect.height) * vb.h + vb.y;
      const nw = vb.w * factor;
      const nh = vb.h * factor;
      setViewBox({
        x: mx - (mx - vb.x) * (nw / vb.w),
        y: my - (my - vb.y) * (nh / vb.h),
        w: nw,
        h: nh,
      });
    },
    [vb]
  );

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (e.button !== 0) return;
      setIsPanning(true);
      panStart.current = { x: e.clientX, y: e.clientY, vx: vb.x, vy: vb.y };
    },
    [vb]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!isPanning) return;
      const rect = svgRef.current?.getBoundingClientRect();
      if (!rect) return;
      const dx = ((e.clientX - panStart.current.x) / rect.width) * vb.w;
      const dy = ((e.clientY - panStart.current.y) / rect.height) * vb.h;
      setViewBox({
        ...vb,
        x: panStart.current.vx - dx,
        y: panStart.current.vy - dy,
      });
    },
    [isPanning, vb]
  );

  const handleMouseUp = useCallback(() => setIsPanning(false), []);

  return (
    <div style={{ position: "relative" }}>
      <svg
        ref={svgRef}
        width="100%"
        height={HEIGHT}
        viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
        style={{
          backgroundColor: colors.plotBg,
          borderRadius: 6,
          border: `1px solid ${colors.border}`,
          cursor: isPanning ? "grabbing" : "grab",
          userSelect: "none",
        }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          handleMouseUp();
          setTooltip(null);
        }}
      >
        {/* Grid lines */}
        {[0.25, 0.5, 0.75].map((frac) => (
          <g key={frac}>
            <line
              x1={WIDTH * frac}
              y1={0}
              x2={WIDTH * frac}
              y2={HEIGHT}
              stroke={colors.gridLine}
              strokeWidth={0.5}
            />
            <line
              x1={0}
              y1={HEIGHT * frac}
              x2={WIDTH}
              y2={HEIGHT * frac}
              stroke={colors.gridLine}
              strokeWidth={0.5}
            />
          </g>
        ))}

        {/* Background points */}
        {points
          .filter((p) => !p.isTopK)
          .map((p) => (
            <circle
              key={p.id}
              cx={toSvgX(p.x)}
              cy={toSvgY(p.y)}
              r={2.5}
              fill={colors.dotGrey}
              opacity={0.4}
              onMouseEnter={(e) => {
                const rect = svgRef.current?.getBoundingClientRect();
                if (rect) {
                  setTooltip({
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top - 10,
                    text: p.preview,
                  });
                }
              }}
              onMouseLeave={() => setTooltip(null)}
              style={{ cursor: "crosshair" }}
            />
          ))}

        {/* Top-K points (orange, sized by similarity) */}
        {points
          .filter((p) => p.isTopK)
          .map((p) => (
            <circle
              key={p.id}
              cx={toSvgX(p.x)}
              cy={toSvgY(p.y)}
              r={4 + p.sim * 8}
              fill="#f97316"
              opacity={0.85}
              stroke="#fff"
              strokeWidth={1}
              onMouseEnter={(e) => {
                const rect = svgRef.current?.getBoundingClientRect();
                if (rect) {
                  setTooltip({
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top - 10,
                    text: `[${p.sim.toFixed(3)}] ${p.preview}`,
                  });
                }
              }}
              onMouseLeave={() => setTooltip(null)}
              style={{ cursor: "crosshair" }}
            />
          ))}

        {/* Query point (glowing blue) */}
        <circle
          cx={toSvgX(queryPoint.x)}
          cy={toSvgY(queryPoint.y)}
          r={12}
          fill={colors.accent}
          opacity={0.25}
        />
        <circle
          cx={toSvgX(queryPoint.x)}
          cy={toSvgY(queryPoint.y)}
          r={7}
          fill={colors.accent}
          stroke="#fff"
          strokeWidth={2}
        />

        {/* Legend */}
        <g transform={`translate(${WIDTH - 155}, 14)`}>
          <rect
            x={0}
            y={0}
            width={145}
            height={70}
            rx={4}
            fill={colors.plotBg}
            opacity={0.9}
            stroke={colors.border}
          />
          <circle cx={14} cy={16} r={5} fill={colors.accent} />
          <text x={26} y={20} fontSize={11} fill={colors.textSecondary}>
            Query
          </text>
          <circle cx={14} cy={34} r={5} fill="#f97316" />
          <text x={26} y={38} fontSize={11} fill={colors.textSecondary}>
            Top-K results
          </text>
          <circle cx={14} cy={52} r={3} fill={colors.dotGrey} opacity={0.6} />
          <text x={26} y={56} fontSize={11} fill={colors.textSecondary}>
            Index chunks
          </text>
        </g>
      </svg>

      {/* Tooltip */}
      {tooltip && (
        <div
          style={{
            position: "absolute",
            left: tooltip.x,
            top: tooltip.y,
            transform: "translate(-50%, -100%)",
            backgroundColor: colors.tooltipBg,
            color: colors.text,
            border: `1px solid ${colors.border}`,
            borderRadius: 4,
            padding: "6px 10px",
            fontSize: 11,
            fontFamily: "monospace",
            maxWidth: 280,
            pointerEvents: "none",
            zIndex: 10,
            whiteSpace: "pre-wrap",
            boxShadow: "0 2px 8px rgba(0,0,0,0.15)",
          }}
        >
          {tooltip.text}
        </div>
      )}
    </div>
  );
}

// --- Histogram (SVG) ---
function Histogram({
  binCounts,
  colors,
}: {
  binCounts: number[];
  colors: ReturnType<typeof useColors>;
}) {
  const bins = useMemo(() => {
    const numBins = binCounts.length;
    const maxCount = Math.max(...binCounts, 1);
    return binCounts.map((c, i) => ({
      x0: i / numBins,
      x1: (i + 1) / numBins,
      count: c,
      height: c / maxCount,
    }));
  }, [binCounts]);

  const W = 500;
  const H = 160;
  const PAD = 30;
  const chartW = W - PAD * 2;
  const chartH = H - PAD - 10;

  return (
    <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`}>
      {/* Axis */}
      <line
        x1={PAD}
        y1={H - PAD}
        x2={W - PAD}
        y2={H - PAD}
        stroke={colors.border}
      />
      {/* Labels */}
      {[0, 0.25, 0.5, 0.75, 1.0].map((v) => (
        <text
          key={v}
          x={PAD + v * chartW}
          y={H - PAD + 14}
          fontSize={10}
          fill={colors.textSecondary}
          textAnchor="middle"
          fontFamily="monospace"
        >
          {v.toFixed(2)}
        </text>
      ))}
      <text
        x={W / 2}
        y={H - 2}
        fontSize={10}
        fill={colors.textSecondary}
        textAnchor="middle"
      >
        Cosine Similarity
      </text>

      {/* Bars */}
      {bins.map((bin, i) => {
        const barW = chartW / bins.length - 1;
        const barH = bin.height * chartH;
        const x = PAD + (i / bins.length) * chartW + 0.5;
        const color = simColor(bin.x0 + 0.02);
        return (
          <rect
            key={i}
            x={x}
            y={H - PAD - barH}
            width={barW}
            height={barH}
            fill={color}
            opacity={0.7}
            rx={1}
          />
        );
      })}
    </svg>
  );
}

// --- Color theme hook ---
function useColors() {
  const theme = useWidgetTheme();
  const isDark = theme === "dark";
  return {
    bg: isDark ? "#0f1117" : "#f8f9fb",
    cardBg: isDark ? "#1a1d27" : "#ffffff",
    text: isDark ? "#e2e4e9" : "#1a1d27",
    textSecondary: isDark ? "#8b8fa3" : "#6b7280",
    border: isDark ? "#2a2d3a" : "#e2e5ea",
    accent: "#3b82f6",
    accentLight: isDark ? "rgba(59,130,246,0.15)" : "rgba(59,130,246,0.08)",
    plotBg: isDark ? "#13151d" : "#f0f2f5",
    gridLine: isDark ? "#1e2130" : "#e0e3e8",
    dotGrey: isDark ? "#555" : "#aaa",
    tooltipBg: isDark ? "#1a1d27" : "#fff",
    tabActive: isDark ? "#1a1d27" : "#fff",
    tabInactive: isDark ? "#0f1117" : "#f0f2f5",
    inputBg: isDark ? "#1a1d27" : "#fff",
  };
}

// --- Main Widget ---
export default function VectorLens() {
  const { props, isPending } = useWidget<Props>();
  const colors = useColors();
  const { callTool: doSearch, isPending: isSearching } = useCallTool("search");

  const [tab, setTab] = useState<Tab>("results");
  const [searchQuery, setSearchQuery] = useState("");
  const [searchK, setSearchK] = useState(5);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());

  // Sync search inputs from props
  useMemo(() => {
    if (!isPending) {
      setSearchQuery(props.query);
      setSearchK(props.k);
    }
  }, [isPending, props.query, props.k]);

  if (isPending) {
    return (
      <McpUseProvider autoSize>
        <div
          style={{
            padding: 40,
            textAlign: "center",
            fontFamily:
              '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace',
            color: "#6b7280",
            backgroundColor: "#f8f9fb",
          }}
        >
          <div style={{ fontSize: 14, marginBottom: 8 }}>
            Embedding query and searching index...
          </div>
          <div
            style={{
              width: 32,
              height: 32,
              border: "3px solid #e2e5ea",
              borderTop: "3px solid #3b82f6",
              borderRadius: "50%",
              margin: "0 auto",
              animation: "spin 1s linear infinite",
            }}
          />
          <style>{`@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`}</style>
        </div>
      </McpUseProvider>
    );
  }

  const handleSearch = () => {
    if (!searchQuery.trim()) return;
    doSearch({ query: searchQuery, k: searchK });
  };

  const handleExport = () => {
    const data = JSON.stringify(props.results, null, 2);
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `vector-lens-results-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const toggleExpand = (id: string) => {
    const next = new Set(expandedIds);
    if (next.has(id)) next.delete(id);
    else next.add(id);
    setExpandedIds(next);
  };

  const tabs: { key: Tab; label: string }[] = [
    { key: "results", label: "Results" },
    { key: "embedding", label: "Embedding Space" },
    { key: "stats", label: "Stats" },
  ];

  const mono: React.CSSProperties = {
    fontFamily: '"SF Mono", "Fira Code", "Cascadia Code", monospace',
  };

  return (
    <McpUseProvider autoSize>
      <div
        style={{
          backgroundColor: colors.bg,
          color: colors.text,
          fontFamily:
            '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
          borderRadius: 8,
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "16px 20px",
            borderBottom: `1px solid ${colors.border}`,
            display: "flex",
            alignItems: "center",
            gap: 16,
            flexWrap: "wrap",
          }}
        >
          <div style={{ flex: "1 1 auto" }}>
            <div
              style={{
                fontSize: 16,
                fontWeight: 700,
                letterSpacing: "-0.01em",
              }}
            >
              Vector Lens
            </div>
            <div
              style={{
                fontSize: 12,
                color: colors.textSecondary,
                marginTop: 2,
                ...mono,
              }}
            >
              {props.datasetName} | {props.stats.indexSize} chunks |{" "}
              {props.modelName}
            </div>
          </div>
          <button
            onClick={handleExport}
            style={{
              padding: "6px 14px",
              fontSize: 12,
              border: `1px solid ${colors.border}`,
              borderRadius: 4,
              backgroundColor: colors.cardBg,
              color: colors.textSecondary,
              cursor: "pointer",
              ...mono,
            }}
          >
            Export results
          </button>
        </div>

        {/* Search bar */}
        <div
          style={{
            padding: "12px 20px",
            borderBottom: `1px solid ${colors.border}`,
            display: "flex",
            gap: 8,
            alignItems: "center",
          }}
        >
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSearch();
            }}
            placeholder="Search query..."
            disabled={isSearching}
            style={{
              flex: 1,
              padding: "8px 12px",
              fontSize: 13,
              border: `1px solid ${colors.border}`,
              borderRadius: 4,
              backgroundColor: colors.inputBg,
              color: colors.text,
              outline: "none",
              ...mono,
            }}
          />
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
              fontSize: 12,
              color: colors.textSecondary,
            }}
          >
            <span style={mono}>k=</span>
            <input
              type="number"
              value={searchK}
              onChange={(e) =>
                setSearchK(Math.max(1, Math.min(50, parseInt(e.target.value) || 5)))
              }
              min={1}
              max={50}
              disabled={isSearching}
              style={{
                width: 48,
                padding: "8px 6px",
                fontSize: 13,
                border: `1px solid ${colors.border}`,
                borderRadius: 4,
                backgroundColor: colors.inputBg,
                color: colors.text,
                textAlign: "center",
                outline: "none",
                ...mono,
              }}
            />
          </div>
          <button
            onClick={handleSearch}
            disabled={isSearching}
            style={{
              padding: "8px 18px",
              fontSize: 13,
              fontWeight: 600,
              border: "none",
              borderRadius: 4,
              backgroundColor: colors.accent,
              color: "#fff",
              cursor: isSearching ? "not-allowed" : "pointer",
              opacity: isSearching ? 0.6 : 1,
            }}
          >
            {isSearching ? "Searching..." : "Search"}
          </button>
        </div>

        {/* Tabs */}
        <div
          style={{
            display: "flex",
            borderBottom: `1px solid ${colors.border}`,
            padding: "0 20px",
          }}
        >
          {tabs.map((t) => (
            <button
              key={t.key}
              onClick={() => setTab(t.key)}
              style={{
                padding: "10px 18px",
                fontSize: 13,
                fontWeight: tab === t.key ? 600 : 400,
                color: tab === t.key ? colors.accent : colors.textSecondary,
                backgroundColor: "transparent",
                border: "none",
                borderBottom:
                  tab === t.key
                    ? `2px solid ${colors.accent}`
                    : "2px solid transparent",
                cursor: "pointer",
                marginBottom: -1,
              }}
            >
              {t.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div style={{ padding: 20 }}>
          {/* RESULTS TAB */}
          {tab === "results" && (
            <div>
              <div
                style={{
                  fontSize: 12,
                  color: colors.textSecondary,
                  marginBottom: 12,
                  ...mono,
                }}
              >
                Showing top {props.results.length} of {props.stats.indexSize}{" "}
                chunks for &quot;{props.query}&quot;
              </div>

              {props.results.map((r, idx) => (
                <div
                  key={r.id}
                  style={{
                    marginBottom: 10,
                    border: `1px solid ${colors.border}`,
                    borderRadius: 6,
                    backgroundColor: colors.cardBg,
                    overflow: "hidden",
                  }}
                >
                  {/* Similarity bar */}
                  <div
                    style={{
                      height: 4,
                      background: `linear-gradient(to right, ${simColor(r.similarity)} ${r.similarity * 100}%, ${colors.bg} ${r.similarity * 100}%)`,
                    }}
                  />

                  <div style={{ padding: "12px 16px" }}>
                    {/* Header row */}
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 10,
                        marginBottom: 8,
                      }}
                    >
                      <span
                        style={{
                          ...mono,
                          fontSize: 12,
                          fontWeight: 700,
                          color: simColor(r.similarity),
                          backgroundColor: simBgColor(r.similarity),
                          padding: "2px 8px",
                          borderRadius: 4,
                        }}
                      >
                        {r.similarity.toFixed(4)}
                      </span>
                      <span
                        style={{
                          ...mono,
                          fontSize: 11,
                          color: colors.textSecondary,
                        }}
                      >
                        #{idx + 1}
                      </span>
                      <span
                        style={{
                          ...mono,
                          fontSize: 11,
                          color: colors.textSecondary,
                        }}
                      >
                        {r.id}
                      </span>
                    </div>

                    {/* Text */}
                    <div
                      onClick={() => toggleExpand(r.id)}
                      style={{
                        fontSize: 13,
                        lineHeight: 1.55,
                        cursor: "pointer",
                        marginBottom: 8,
                        color: colors.text,
                      }}
                    >
                      {highlightText(
                        r.text,
                        props.query,
                        200,
                        expandedIds.has(r.id),
                        colors.accent
                      )}
                    </div>

                    {/* Metadata */}
                    <div
                      style={{
                        fontSize: 11,
                        color: colors.textSecondary,
                        display: "flex",
                        gap: 12,
                        flexWrap: "wrap",
                      }}
                    >
                      <span style={{ fontWeight: 600 }}>{r.source}</span>
                      <span style={mono}>{r.year}</span>
                      <span>{r.authors}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* EMBEDDING SPACE TAB */}
          {tab === "embedding" && (
            <div>
              <div
                style={{
                  fontSize: 12,
                  color: colors.textSecondary,
                  marginBottom: 12,
                  ...mono,
                }}
              >
                2D projection ({props.spacePoints.length} points) | Scroll to
                zoom, drag to pan
              </div>
              <ScatterPlot
                points={props.spacePoints}
                queryPoint={props.queryPoint}
                colors={colors}
              />
            </div>
          )}

          {/* STATS TAB */}
          {tab === "stats" && (
            <div>
              {/* Stat cards */}
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(150px, 1fr))",
                  gap: 10,
                  marginBottom: 20,
                }}
              >
                {[
                  {
                    label: "Index Size",
                    value: String(props.stats.indexSize),
                    unit: "chunks",
                  },
                  {
                    label: "Embedding Dim",
                    value: String(props.stats.embeddingDim),
                    unit: "dims",
                  },
                  {
                    label: "Distance Metric",
                    value: props.stats.distanceMetric,
                    unit: "",
                  },
                  {
                    label: "Query Latency",
                    value: String(props.stats.latencyMs),
                    unit: "ms",
                  },
                  {
                    label: "Top-1 Similarity",
                    value: props.stats.top1Sim.toFixed(4),
                    unit: "",
                  },
                  {
                    label: "Top-K Sim Gap",
                    value: props.stats.topKSimGap.toFixed(4),
                    unit: "",
                  },
                ].map((stat) => (
                  <div
                    key={stat.label}
                    style={{
                      padding: 14,
                      border: `1px solid ${colors.border}`,
                      borderRadius: 6,
                      backgroundColor: colors.cardBg,
                    }}
                  >
                    <div
                      style={{
                        fontSize: 11,
                        color: colors.textSecondary,
                        marginBottom: 4,
                        textTransform: "uppercase",
                        letterSpacing: "0.05em",
                      }}
                    >
                      {stat.label}
                    </div>
                    <div
                      style={{
                        fontSize: 22,
                        fontWeight: 700,
                        ...mono,
                        color: colors.text,
                      }}
                    >
                      {stat.value}
                      {stat.unit && (
                        <span
                          style={{
                            fontSize: 11,
                            fontWeight: 400,
                            color: colors.textSecondary,
                            marginLeft: 4,
                          }}
                        >
                          {stat.unit}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>

              {/* Similarity distribution histogram */}
              <div
                style={{
                  border: `1px solid ${colors.border}`,
                  borderRadius: 6,
                  backgroundColor: colors.cardBg,
                  padding: 16,
                }}
              >
                <div
                  style={{
                    fontSize: 13,
                    fontWeight: 600,
                    marginBottom: 10,
                  }}
                >
                  Similarity Score Distribution
                </div>
                <div
                  style={{
                    fontSize: 11,
                    color: colors.textSecondary,
                    marginBottom: 8,
                    ...mono,
                  }}
                >
                  All {props.stats.indexSize} chunks scored against query
                </div>
                <Histogram binCounts={props.histBins} colors={colors} />
              </div>
            </div>
          )}
        </div>
      </div>
    </McpUseProvider>
  );
}
