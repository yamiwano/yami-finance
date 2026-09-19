"use client";

import { useEffect, useRef, useState } from "react";
import { api, endpoints } from "@/lib/api";
import type { Activity, Health, Signal } from "@/lib/types";

export function useLiveSignals(params: Record<string, string | number | undefined> = {}) {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [activity, setActivity] = useState<Activity[]>([]);
  const [connected, setConnected] = useState(false);
  const [scans, setScans] = useState(0);
  const [health, setHealth] = useState<Health | null>(null);
  const key = JSON.stringify(params);
  const paramsRef = useRef(params);
  paramsRef.current = params;

  async function refresh() {
    const [s, a, h] = await Promise.all([api.signals(paramsRef.current), api.activity(), api.health().catch(() => null)]);
    setSignals(s);
    setActivity(a);
    if (h) {
      setScans(h.scans);
      setHealth(h);
    }
  }

  useEffect(() => {
    let ws: WebSocket | null = null;
    let cancelled = false;
    refresh().catch(() => undefined);

    endpoints()
      .then(({ ws: url }) => {
        if (cancelled) return;
        ws = new WebSocket(url);
        ws.onopen = () => setConnected(true);
        ws.onclose = () => setConnected(false);
        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data) as { event: string; payload: Signal };
            if (msg.event === "scanner.tick") {
              setScans((n) => n + 1);
              return;
            }
            if (msg.event === "signal.created" || msg.event === "signal.updated") {
              const incoming = msg.payload;
              setSignals((prev) => {
                const i = prev.findIndex((x) => x.id === incoming.id);
                if (i >= 0) {
                  const next = prev.slice();
                  next[i] = { ...next[i], ...incoming };
                  return next.sort((a, b) => b.score - a.score);
                }
                return [incoming, ...prev].sort((a, b) => b.score - a.score);
              });
            }
          } catch {
            /* ignore malformed frames */
          }
        };
      })
      .catch(() => undefined);

    const t = setInterval(() => refresh().catch(() => undefined), 12000);
    return () => {
      cancelled = true;
      ws?.close();
      clearInterval(t);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return { signals, activity, connected, scans, health, refresh };
}
