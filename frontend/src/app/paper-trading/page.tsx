"use client";

import { useEffect, useState } from "react";

type PaperSnapshot = {
  status: { enabled: boolean; active: boolean; last_cycle_at: string | null; error: string | null };
  account: {
    starting_capital: number;
    cash: number;
    equity: number;
    realized_pnl: number;
    total_fees: number;
    total_slippage: number;
    open_positions: number;
  };
  positions: Array<{
    symbol: string;
    direction: string;
    entry_ts: string;
    entry_price: number;
    weight: number;
    status: string;
  }>;
  signals: Array<{
    ts: string;
    symbol: string;
    probability: number;
    rank: number;
    status: string;
    model_version: string;
  }>;
  performance: {
    n_trades: number;
    win_rate: number | null;
    avg_trade: number | null;
    median_trade: number | null;
    cumulative_return: number;
  };
};

export default function PaperTradingPage() {
  const [data, setData] = useState<PaperSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/paper-trading")
      .then((r) => r.json())
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) return <div className="p-6 text-red-400">Error: {error}</div>;
  if (!data) return <div className="p-6">Loading paper trading…</div>;

  const { status, account, positions, signals, performance } = data;

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Paper Trading</h1>
        <div className="text-sm px-3 py-1 rounded bg-amber-900/40 text-amber-200">
          PAPER TRADE — Not a live order
        </div>
      </div>

      <div className="text-sm text-zinc-400">
        Status: {status.enabled ? (status.active ? "ACTIVE" : "PAUSED") : "DISABLED"}
        {status.last_cycle_at && ` · Last cycle ${status.last_cycle_at}`}
        {status.error && ` · Error: ${status.error}`}
      </div>

      <section className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Metric label="Starting capital" value={`$${account.starting_capital.toFixed(2)}`} />
        <Metric label="Equity" value={`$${account.equity.toFixed(2)}`} />
        <Metric label="Cash" value={`$${account.cash.toFixed(2)}`} />
        <Metric label="Realized P&L" value={`$${account.realized_pnl.toFixed(2)}`} />
        <Metric label="Total fees" value={`$${account.total_fees.toFixed(2)}`} />
        <Metric label="Open positions" value={String(account.open_positions)} />
        <Metric label="Trades" value={String(performance.n_trades)} />
        <Metric
          label="Win rate"
          value={performance.win_rate == null ? "—" : `${(performance.win_rate * 100).toFixed(1)}%`}
        />
      </section>

      <section>
        <h2 className="text-lg font-medium mb-2">Current positions</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-zinc-400">
              <tr>
                <th className="text-left py-1">Symbol</th>
                <th className="text-left py-1">Direction</th>
                <th className="text-left py-1">Entry</th>
                <th className="text-left py-1">Weight</th>
                <th className="text-left py-1">Status</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((p) => (
                <tr key={`${p.symbol}-${p.entry_ts}`} className="border-t border-zinc-800">
                  <td className="py-1">{p.symbol}</td>
                  <td className="py-1">{p.direction}</td>
                  <td className="py-1">{p.entry_price}</td>
                  <td className="py-1">{(p.weight * 100).toFixed(1)}%</td>
                  <td className="py-1">{p.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section>
        <h2 className="text-lg font-medium mb-2">Latest signals</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-zinc-400">
              <tr>
                <th className="text-left py-1">Timestamp</th>
                <th className="text-left py-1">Symbol</th>
                <th className="text-left py-1">Probability</th>
                <th className="text-left py-1">Rank</th>
                <th className="text-left py-1">Status</th>
              </tr>
            </thead>
            <tbody>
              {signals.map((s) => (
                <tr key={`${s.symbol}-${s.ts}`} className="border-t border-zinc-800">
                  <td className="py-1">{s.ts}</td>
                  <td className="py-1">{s.symbol}</td>
                  <td className="py-1">{s.probability.toFixed(3)}</td>
                  <td className="py-1">{s.rank}</td>
                  <td className="py-1">{s.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-zinc-800 p-3">
      <div className="text-xs text-zinc-400">{label}</div>
      <div className="text-lg font-medium">{value}</div>
    </div>
  );
}
