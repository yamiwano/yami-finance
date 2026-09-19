"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { BrandMark } from "./BrandMark";

const NAV = [
  { href: "/", label: "Dashboard" },
  { href: "/scanner", label: "Scanner" },
  { href: "/history", label: "History" },
  { href: "/performance", label: "Performance" },
  { href: "/watchlist", label: "Watchlist" },
  { href: "/settings", label: "Settings" },
];

export function Shell({ children }: { children: React.ReactNode }) {
  const path = usePathname();
  return (
    <div className="min-h-screen flex">
      <aside className="w-56 shrink-0 border-r border-line bg-panel flex flex-col">
        <div className="px-4 py-5 border-b border-line flex gap-3 items-start">
          <BrandMark size={40} />
          <div className="min-w-0">
            <div className="text-[11px] tracking-[0.22em] text-accent uppercase">YF</div>
            <div className="text-base font-semibold leading-tight mt-0.5">Yami Financier</div>
            <div className="text-[11px] text-mute mt-1 leading-snug">Binance USDT spot. Analytical support, not a prediction engine.</div>
          </div>
        </div>
        <nav className="p-2 flex-1">
          {NAV.map((n) => {
            const active = n.href === "/" ? path === "/" : path.startsWith(n.href);
            return (
              <Link
                key={n.href}
                href={n.href}
                className={`block px-3 py-2 rounded text-[13px] mb-0.5 ${
                  active ? "bg-accent/15 text-accent" : "text-ink/80 hover:bg-line/60"
                }`}
              >
                {n.label}
              </Link>
            );
          })}
        </nav>
        <div className="p-4 text-[10px] text-mute border-t border-line leading-relaxed">
          Signals are hypothetical setups. Levels are not guaranteed. Do not treat this as investment advice.
        </div>
      </aside>
      <main className="flex-1 min-w-0">{children}</main>
    </div>
  );
}
