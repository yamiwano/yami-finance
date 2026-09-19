import type { Activity, ChartPayload, Health, Performance, Research, ScannerRow, Settings, Signal, WatchItem } from "./types";

declare global {
  interface Window {
    yamiDesktop?: { apiBase: string; wsBase: string };
  }
}

type Endpoints = { api: string; ws: string };

let cached: Endpoints | null = null;
let inflight: Promise<Endpoints> | null = null;

export async function endpoints(): Promise<Endpoints> {
  if (typeof window !== "undefined" && window.yamiDesktop?.apiBase) {
    return { api: window.yamiDesktop.apiBase, ws: window.yamiDesktop.wsBase };
  }
  if (cached) return cached;
  if (!inflight) {
    inflight = (async () => {
      if (typeof window !== "undefined") {
        try {
          const r = await fetch("/runtime", { cache: "no-store" });
          if (r.ok) {
            const j = await r.json();
            cached = { api: j.apiBase, ws: j.wsBase };
            return cached;
          }
        } catch {
          /* browser fallback */
        }
      }
      cached = {
        api: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
        ws: process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws",
      };
      return cached;
    })();
  }
  return inflight;
}

export function wsBase(): string {
  if (typeof window !== "undefined" && window.yamiDesktop?.wsBase) return window.yamiDesktop.wsBase;
  return process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000/ws";
}

async function get<T>(path: string): Promise<T> {
  const { api } = await endpoints();
  const r = await fetch(`${api}${path}`, { cache: "no-store" });
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}

async function send<T>(path: string, method: string, body?: unknown): Promise<T> {
  const { api } = await endpoints();
  const r = await fetch(`${api}${path}`, {
    method,
    headers: { "Content-Type": "application/json" },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) throw new Error(`${r.status} ${path}`);
  return r.json();
}

export const api = {
  health: () => get<Health>("/api/health"),
  meta: () => get<{ strategies: { id: string; label: string }[]; timeframes: string[]; disclaimer: string; data_source?: string }>("/api/meta"),
  activity: () => get<Activity[]>("/api/activity"),
  quotes: () => get<Record<string, { symbol: string; price: number; change_pct: number }>>("/api/quotes"),
  signals: (params: Record<string, string | number | undefined> = {}) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== "") q.set(k, String(v));
    });
    return get<Signal[]>(`/api/signals?${q.toString()}`);
  },
  signal: (id: string) => get<Signal>(`/api/signals/${id}`),
  signalChart: (id: string) => get<ChartPayload>(`/api/signals/${id}/chart`),
  scanner: (params: Record<string, string | number | undefined> = {}) => {
    const q = new URLSearchParams();
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined && v !== "") q.set(k, String(v));
    });
    return get<ScannerRow[]>(`/api/scanner?${q.toString()}`);
  },
  history: (status?: string) => get<Signal[]>(`/api/history${status ? `?status=${status}` : ""}`),
  performance: () => get<Performance>("/api/performance"),
  resetPerformance: () => send<Performance>("/api/performance/reset", "POST"),
  research: () => get<Research>("/api/research"),
  trainResearch: () => send<{ ok: boolean; status: string; message?: string }>("/api/research/train", "POST"),
  settings: () => get<Settings>("/api/settings"),
  saveSettings: (body: Partial<Settings>) => send<Settings>("/api/settings", "PUT", body),
  watchlist: () => get<WatchItem[]>("/api/watchlist"),
  addWatch: (symbol: string, notes = "") => send<{ ok: boolean }>("/api/watchlist", "POST", { symbol, notes }),
  removeWatch: (id: string) => send<{ ok: boolean }>(`/api/watchlist/${id}`, "DELETE"),
};
