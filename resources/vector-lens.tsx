import { useState, useMemo, useRef, useCallback, useEffect } from "react";
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
      topics: z.array(z.string()),
      similarity: z.number(),
      denseScore: z.number(),
      lexicalScore: z.number(),
      topicOverlap: z.number(),
    })
  ),
  spacePoints: z.array(
    z.object({
      id: z.string(),
      x: z.number(),
      y: z.number(),
      sim: z.number(),
      isTopK: z.boolean(),
      source: z.string(),
      year: z.number(),
      topics: z.array(z.string()),
      rank: z.number().nullable(),
      preview: z.string(),
    })
  ),
  queryPoint: z.object({ x: z.number(), y: z.number(), topics: z.array(z.string()) }),
  stats: z.object({
    indexSize: z.number(),
    embeddingDim: z.number(),
    distanceMetric: z.string(),
    latencyMs: z.number(),
    top1Sim: z.number(),
    topKSimGap: z.number(),
    scoringMode: z.string(),
    matchedTopics: z.array(z.string()),
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
  selectedId,
  onSelect,
  colors,
}: {
  points: Props["spacePoints"];
  queryPoint: Props["queryPoint"];
  selectedId: string | null;
  onSelect: (id: string) => void;
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

  useEffect(() => {
    if (viewBox.w === 0) setViewBox(bounds);
  }, [bounds, viewBox.w]);

  const vb = viewBox.w > 0 ? viewBox : bounds;
  const WIDTH = 600;
  const HEIGHT = 400;
  const topKPoints = useMemo(() => points.filter((p) => p.isTopK), [points]);
  const selectedPoint = useMemo(
    () => points.find((point) => point.id === selectedId) ?? null,
    [points, selectedId]
  );

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

  const resetView = useCallback(() => setViewBox(bounds), [bounds]);

  const focusTopK = useCallback(() => {
    const focusPoints = [...topKPoints, { ...queryPoint, id: "query" }];
    const allX = focusPoints.map((point) => point.x);
    const allY = focusPoints.map((point) => point.y);
    const minX = Math.min(...allX);
    const maxX = Math.max(...allX);
    const minY = Math.min(...allY);
    const maxY = Math.max(...allY);
    const padX = (maxX - minX) * 0.45 || 1.2;
    const padY = (maxY - minY) * 0.45 || 1.2;
    setViewBox({
      x: minX - padX,
      y: minY - padY,
      w: maxX - minX + padX * 2,
      h: maxY - minY + padY * 2,
    });
  }, [queryPoint, topKPoints]);

  return (
    <div style={{ position: "relative" }}>
      <div
        style={{
          position: "absolute",
          top: 12,
          left: 12,
          display: "flex",
          gap: 8,
          zIndex: 2,
        }}
      >
        {[
          { label: "Reset view", onClick: resetView },
          { label: "Focus top-k", onClick: focusTopK },
        ].map((action) => (
          <button
            key={action.label}
            onClick={action.onClick}
            style={{
              border: `1px solid ${colors.border}`,
              backgroundColor: colors.toolbarBg,
              color: colors.text,
              borderRadius: 999,
              padding: "6px 10px",
              fontSize: 11,
              cursor: "pointer",
              backdropFilter: "blur(10px)",
            }}
          >
            {action.label}
          </button>
        ))}
      </div>
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
          boxShadow: "inset 0 1px 0 rgba(255,255,255,0.06)",
        }}
        onWheel={handleWheel}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={() => {
          handleMouseUp();
          setTooltip(null);
        }}
        onDoubleClick={resetView}
      >
        <defs>
          <radialGradient id="vector-lens-glow" cx="50%" cy="50%" r="70%">
            <stop offset="0%" stopColor={colors.plotGlow} stopOpacity="0.32" />
            <stop offset="100%" stopColor={colors.plotGlow} stopOpacity="0" />
          </radialGradient>
        </defs>

        <rect x={0} y={0} width={WIDTH} height={HEIGHT} fill="url(#vector-lens-glow)" />

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

        {topKPoints.map((point) => (
          <line
            key={`link-${point.id}`}
            x1={toSvgX(queryPoint.x)}
            y1={toSvgY(queryPoint.y)}
            x2={toSvgX(point.x)}
            y2={toSvgY(point.y)}
            stroke={colors.linkLine}
            strokeDasharray="5 5"
            strokeWidth={point.rank === 1 ? 1.8 : 1}
            opacity={point.rank === 1 ? 0.8 : 0.45}
          />
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
              onClick={() => onSelect(p.id)}
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
              fill={p.rank === 1 ? colors.hot : colors.warm}
              opacity={0.85}
              stroke={selectedId === p.id ? colors.selection : "#fff"}
              strokeWidth={selectedId === p.id ? 2.5 : 1.25}
              onClick={() => onSelect(p.id)}
              onMouseEnter={(e) => {
                const rect = svgRef.current?.getBoundingClientRect();
                if (rect) {
                  setTooltip({
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top - 10,
                    text: `#${p.rank ?? "?"} [${p.sim.toFixed(3)}] ${p.source}\n${p.preview}`,
                  });
                }
              }}
              onMouseLeave={() => setTooltip(null)}
              style={{ cursor: "crosshair" }}
            />
          ))}

        {selectedPoint && (
          <circle
            cx={toSvgX(selectedPoint.x)}
            cy={toSvgY(selectedPoint.y)}
            r={14}
            fill="none"
            stroke={colors.selection}
            strokeWidth={2}
            opacity={0.95}
          />
        )}

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

        {topKPoints.slice(0, 3).map((point) => (
          <text
            key={`label-${point.id}`}
            x={toSvgX(point.x) + 8}
            y={toSvgY(point.y) - 8}
            fontSize={11}
            fontWeight={700}
            fill={colors.text}
            style={{ pointerEvents: "none" }}
          >
            #{point.rank}
          </text>
        ))}

        {/* Legend */}
        <g transform={`translate(${WIDTH - 168}, 16)`}>
          <rect
            x={0}
            y={0}
            width={154}
            height={86}
            rx={10}
            fill={colors.toolbarBg}
            opacity={0.95}
            stroke={colors.border}
          />
          <circle cx={16} cy={18} r={5} fill={colors.accent} />
          <text x={28} y={22} fontSize={11} fill={colors.textSecondary}>
            Query
          </text>
          <circle cx={16} cy={38} r={5} fill={colors.warm} />
          <text x={28} y={42} fontSize={11} fill={colors.textSecondary}>
            Top-K results
          </text>
          <circle cx={16} cy={58} r={3} fill={colors.dotGrey} opacity={0.6} />
          <text x={28} y={62} fontSize={11} fill={colors.textSecondary}>
            Index chunks
          </text>
          <line
            x1={16}
            y1={74}
            x2={28}
            y2={74}
            stroke={colors.linkLine}
            strokeDasharray="4 4"
          />
          <text x={34} y={78} fontSize={11} fill={colors.textSecondary}>
            Query-to-neighbor links
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
    bg: isDark ? "#0d1018" : "#f4f6fb",
    cardBg: isDark ? "#171b26" : "#ffffff",
    text: isDark ? "#e5e9f5" : "#162033",
    textSecondary: isDark ? "#93a0ba" : "#637189",
    border: isDark ? "#273149" : "#d9e0ef",
    accent: "#2f7df6",
    accentLight: isDark ? "rgba(47,125,246,0.18)" : "rgba(47,125,246,0.08)",
    plotBg: isDark ? "#101624" : "#edf3ff",
    plotGlow: isDark ? "#2f7df6" : "#69a5ff",
    gridLine: isDark ? "#20283a" : "#d5deef",
    dotGrey: isDark ? "#5f6c85" : "#98a4b6",
    tooltipBg: isDark ? "#141a27" : "#ffffff",
    tabActive: isDark ? "#171b26" : "#ffffff",
    tabInactive: isDark ? "#101624" : "#edf3ff",
    inputBg: isDark ? "#171b26" : "#ffffff",
    toolbarBg: isDark ? "rgba(23,27,38,0.86)" : "rgba(255,255,255,0.88)",
    warm: "#fb923c",
    hot: "#f97316",
    selection: "#14b8a6",
    linkLine: isDark ? "rgba(99,179,237,0.55)" : "rgba(59,130,246,0.42)",
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
  const [selectedPointId, setSelectedPointId] = useState<string | null>(null);

  useEffect(() => {
    if (!isPending) {
      setSearchQuery(props.query);
      setSearchK(props.k);
      setSelectedPointId(props.results[0]?.id ?? null);
    }
  }, [isPending, props.query, props.k, props.results]);

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
            padding: "18px 20px",
            borderBottom: `1px solid ${colors.border}`,
            display: "flex",
            alignItems: "center",
            gap: 16,
            flexWrap: "wrap",
            background:
              "linear-gradient(135deg, rgba(47,125,246,0.08), rgba(20,184,166,0.05) 45%, transparent 80%)",
          }}
        >
          <div style={{ flex: "1 1 auto" }}>
            <div
              style={{
                fontSize: 18,
                fontWeight: 700,
                letterSpacing: "-0.03em",
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
              {props.datasetName} | {props.stats.indexSize} chunks | {props.modelName}
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
                      <button
                        onClick={() => {
                          setSelectedPointId(r.id);
                          setTab("embedding");
                        }}
                        style={{
                          marginLeft: "auto",
                          border: `1px solid ${colors.border}`,
                          backgroundColor:
                            selectedPointId === r.id ? colors.accentLight : colors.cardBg,
                          color: colors.textSecondary,
                          borderRadius: 999,
                          padding: "4px 10px",
                          fontSize: 11,
                          cursor: "pointer",
                        }}
                      >
                        Locate in space
                      </button>
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
                        alignItems: "center",
                      }}
                    >
                      <span style={{ fontWeight: 600 }}>{r.source}</span>
                      <span style={mono}>{r.year}</span>
                      <span>{r.authors}</span>
                      {r.topics.map((topic) => (
                        <span
                          key={`${r.id}-${topic}`}
                          style={{
                            border: `1px solid ${colors.border}`,
                            borderRadius: 999,
                            padding: "2px 8px",
                            backgroundColor: colors.bg,
                          }}
                        >
                          {topic}
                        </span>
                      ))}
                    </div>

                    <div
                      style={{
                        marginTop: 10,
                        display: "flex",
                        gap: 8,
                        flexWrap: "wrap",
                      }}
                    >
                      {[
                        `dense ${r.denseScore.toFixed(3)}`,
                        `lexical ${r.lexicalScore.toFixed(3)}`,
                        `topic ${r.topicOverlap.toFixed(3)}`,
                      ].map((chip) => (
                        <span
                          key={`${r.id}-${chip}`}
                          style={{
                            ...mono,
                            fontSize: 11,
                            color: colors.textSecondary,
                            backgroundColor: colors.bg,
                            border: `1px solid ${colors.border}`,
                            borderRadius: 999,
                            padding: "3px 8px",
                          }}
                        >
                          {chip}
                        </span>
                      ))}
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
                2D semantic projection ({props.spacePoints.length} points) | Scroll to zoom,
                drag to pan, double-click to reset
              </div>
              <ScatterPlot
                points={props.spacePoints}
                queryPoint={props.queryPoint}
                selectedId={selectedPointId}
                onSelect={setSelectedPointId}
                colors={colors}
              />
              {selectedPointId && (
                <div
                  style={{
                    marginTop: 14,
                    border: `1px solid ${colors.border}`,
                    borderRadius: 10,
                    backgroundColor: colors.cardBg,
                    padding: 16,
                    boxShadow: "0 10px 30px rgba(15, 23, 42, 0.06)",
                  }}
                >
                  {(() => {
                    const point = props.spacePoints.find((entry) => entry.id === selectedPointId);
                    if (!point) return null;
                    return (
                      <>
                        <div
                          style={{
                            display: "flex",
                            gap: 10,
                            alignItems: "center",
                            flexWrap: "wrap",
                            marginBottom: 8,
                          }}
                        >
                          <span
                            style={{
                              ...mono,
                              color: point.isTopK ? colors.hot : colors.textSecondary,
                              fontWeight: 700,
                            }}
                          >
                            {point.isTopK ? `Top-${point.rank}` : "Indexed chunk"}
                          </span>
                          <span style={{ fontWeight: 700 }}>{point.source}</span>
                          <span style={{ ...mono, color: colors.textSecondary }}>{point.year}</span>
                        </div>
                        <div
                          style={{
                            fontSize: 13,
                            lineHeight: 1.6,
                            color: colors.text,
                            marginBottom: 10,
                          }}
                        >
                          {point.preview}
                        </div>
                        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                          {point.topics.map((topic) => (
                            <span
                              key={`${point.id}-${topic}`}
                              style={{
                                fontSize: 11,
                                color: colors.textSecondary,
                                border: `1px solid ${colors.border}`,
                                borderRadius: 999,
                                padding: "3px 8px",
                                backgroundColor: colors.bg,
                              }}
                            >
                              {topic}
                            </span>
                          ))}
                        </div>
                      </>
                    );
                  })()}
                </div>
              )}
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

              <div
                style={{
                  border: `1px solid ${colors.border}`,
                  borderRadius: 6,
                  backgroundColor: colors.cardBg,
                  padding: 16,
                  marginBottom: 14,
                }}
              >
                <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
                  Retrieval Mix
                </div>
                <div style={{ fontSize: 12, color: colors.textSecondary, ...mono }}>
                  {props.stats.scoringMode}
                </div>
                {props.stats.matchedTopics.length > 0 && (
                  <div
                    style={{
                      display: "flex",
                      gap: 8,
                      flexWrap: "wrap",
                      marginTop: 10,
                    }}
                  >
                    {props.stats.matchedTopics.map((topic) => (
                      <span
                        key={`matched-${topic}`}
                        style={{
                          fontSize: 11,
                          color: colors.textSecondary,
                          border: `1px solid ${colors.border}`,
                          borderRadius: 999,
                          padding: "3px 8px",
                          backgroundColor: colors.bg,
                        }}
                      >
                        query topic: {topic}
                      </span>
                    ))}
                  </div>
                )}
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
