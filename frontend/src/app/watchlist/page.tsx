"use client";

import { useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { WatchItem } from "@/lib/types";
import { Panel } from "@/components/ui";
import { fmtPct, fmtPx } from "@/lib/format";

export default function WatchlistPage() {
  const [items, setItems] = useState<WatchItem[]>([]);
  const [symbol, setSymbol] = useState("");
  const [notes, setNotes] = useState("");

  async function load() {
    setItems(await api.watchlist());
  }

  useEffect(() => {
    load().catch(() => undefined);
    const t = setInterval(() => load().catch(() => undefined), 8000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="p-4 space-y-4">
      <header>
        <div className="text-[11px] uppercase tracking-[0.18em] text-mute">Saved pairs</div>
        <h1 className="text-xl font-semibold">Watchlist</h1>
      </header>
      <form
        className="flex gap-2 items-end"
        onSubmit={async (e) => {
          e.preventDefault();
          if (!symbol.trim()) return;
          await api.addWatch(symbol.trim().toUpperCase(), notes);
          setSymbol("");
          setNotes("");
          await load();
        }}
      >
        <label className="text-[11px] text-mute">
          Symbol
          <input className="block bg-panel border border-line rounded px-2 py-1 text-xs mt-1 w-32" value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="BTCUSDT" />
        </label>
        <label className="text-[11px] text-mute flex-1">
          Notes
          <input className="block bg-panel border border-line rounded px-2 py-1 text-xs mt-1 w-full" value={notes} onChange={(e) => setNotes(e.target.value)} />
        </label>
        <button className="text-xs bg-accent/20 border border-accent/40 rounded px-3 py-1.5">Add</button>
      </form>
      <Panel>
        <table className="w-full text-xs">
          <thead className="text-mute text-[10px] uppercase">
            <tr>
              <th className="text-left px-3 py-2">Symbol</th>
              <th className="text-left px-2 py-2">Name</th>
              <th className="text-right px-2 py-2">Price</th>
              <th className="text-right px-2 py-2">Chg</th>
              <th className="text-left px-2 py-2">Notes</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {items.length === 0 && (
              <tr>
                <td colSpan={6} className="px-3 py-8 text-center text-mute">
                  Empty. Save a Binance USDT pair from a signal or add it above (e.g. BTCUSDT).
                </td>
              </tr>
            )}
            {items.map((it) => (
              <tr key={it.id} className="border-t border-line">
                <td className="px-3 py-2 font-mono">{it.symbol}</td>
                <td className="px-2 py-2 text-mute">{it.name}</td>
                <td className="px-2 py-2 text-right font-mono">{fmtPx(it.price)}</td>
                <td className={`px-2 py-2 text-right font-mono ${(it.change_pct || 0) >= 0 ? "text-long" : "text-short"}`}>{fmtPct(it.change_pct)}</td>
                <td className="px-2 py-2">{it.notes}</td>
                <td className="px-3 py-2 text-right">
                  <button
                    className="text-short text-[11px]"
                    onClick={async () => {
                      await api.removeWatch(it.id);
                      await load();
                    }}
                  >
                    Remove
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </Panel>
    </div>
  );
}
