"use client";

import { useEffect, useRef } from "react";
import { createChart, ColorType, CrosshairMode, LineStyle, type IChartApi } from "lightweight-charts";
import type { ChartPayload, Signal } from "@/lib/types";

function toUnix(iso: string): number {
  return Math.floor(new Date(iso).getTime() / 1000);
}

export function RadarChart({ chart, signal }: { chart: ChartPayload; signal?: Signal | null }) {
  const candleRef = useRef<HTMLDivElement>(null);
  const rsiRef = useRef<HTMLDivElement>(null);
  const c1 = useRef<IChartApi | null>(null);
  const c2 = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!candleRef.current || !rsiRef.current) return;
    const common = {
      layout: { background: { type: ColorType.Solid, color: "#0e121b" }, textColor: "#6b778c" },
      grid: { vertLines: { color: "#1c2433" }, horzLines: { color: "#1c2433" } },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#1c2433" },
      timeScale: { borderColor: "#1c2433", timeVisible: true, secondsVisible: false },
    };

    const main = createChart(candleRef.current, { ...common, height: 380 });
    const rsi = createChart(rsiRef.current, { ...common, height: 120 });
    c1.current = main;
    c2.current = rsi;

    const candles = main.addCandlestickSeries({
      upColor: "#3dd68c",
      downColor: "#f0616d",
      borderUpColor: "#3dd68c",
      borderDownColor: "#f0616d",
      wickUpColor: "#3dd68c",
      wickDownColor: "#f0616d",
    });
    const vol = main.addHistogramSeries({
      priceFormat: { type: "volume" },
      priceScaleId: "vol",
    });
    main.priceScale("vol").applyOptions({ scaleMargins: { top: 0.78, bottom: 0 } });

    const ema9 = main.addLineSeries({ color: "#4d9fff", lineWidth: 1, title: "EMA9" });
    const ema21 = main.addLineSeries({ color: "#e7c547", lineWidth: 1, title: "EMA21" });
    const ema50 = main.addLineSeries({ color: "#c084fc", lineWidth: 1, title: "EMA50" });
    const vwap = main.addLineSeries({ color: "#fb923c", lineWidth: 1, lineStyle: LineStyle.Dotted, title: "VWAP" });
    const rsiSeries = rsi.addLineSeries({ color: "#4d9fff", lineWidth: 2, title: "RSI" });
    rsi.addLineSeries({ color: "#6b778c", lineWidth: 1, lineStyle: LineStyle.Dotted }).setData([
      { time: toUnix(chart.candles[0].ts) as never, value: 70 },
      { time: toUnix(chart.candles[chart.candles.length - 1].ts) as never, value: 70 },
    ]);
    rsi.addLineSeries({ color: "#6b778c", lineWidth: 1, lineStyle: LineStyle.Dotted }).setData([
      { time: toUnix(chart.candles[0].ts) as never, value: 30 },
      { time: toUnix(chart.candles[chart.candles.length - 1].ts) as never, value: 30 },
    ]);

    const candleData = chart.candles.map((b) => ({
      time: toUnix(b.ts) as never,
      open: b.open,
      high: b.high,
      low: b.low,
      close: b.close,
    }));
    candles.setData(candleData);
    vol.setData(
      chart.candles.map((b) => ({
        time: toUnix(b.ts) as never,
        value: b.volume,
        color: b.close >= b.open ? "rgba(61,214,140,0.35)" : "rgba(240,97,109,0.35)",
      }))
    );
    const line = (arr: (number | null)[]) =>
      chart.candles
        .map((b, i) => (arr[i] == null ? null : { time: toUnix(b.ts) as never, value: arr[i] as number }))
        .filter((x): x is { time: never; value: number } => x != null);
    ema9.setData(line(chart.ema9));
    ema21.setData(line(chart.ema21));
    ema50.setData(line(chart.ema50));
    vwap.setData(line(chart.vwap));
    rsiSeries.setData(line(chart.rsi));

    if (signal) {
      const mk = (price: number, color: string, title: string) =>
        candles.createPriceLine({ price, color, lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: true, title });
      mk(signal.entry_low, "#4d9fff", "Entry L");
      mk(signal.entry_high, "#4d9fff", "Entry H");
      mk(signal.stop, "#f0616d", "Stop");
      mk(signal.target_1, "#3dd68c", "T1");
      mk(signal.target_2, "#3dd68c", "T2");
      mk(signal.invalidation, "#e7c547", "Inv");
      const t = toUnix(signal.detected_at);
      candles.setMarkers([
        {
          time: t as never,
          position: signal.direction === "SHORT" ? "aboveBar" : "belowBar",
          color: signal.direction === "SHORT" ? "#f0616d" : signal.direction === "LONG" ? "#3dd68c" : "#e7c547",
          shape: signal.direction === "SHORT" ? "arrowDown" : "arrowUp",
          text: signal.direction,
        },
      ]);
    }

    main.timeScale().fitContent();
    const sync = (source: IChartApi, target: IChartApi) => {
      source.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range) target.timeScale().setVisibleLogicalRange(range);
      });
    };
    sync(main, rsi);
    sync(rsi, main);

    const ro = new ResizeObserver(() => {
      if (candleRef.current) main.applyOptions({ width: candleRef.current.clientWidth });
      if (rsiRef.current) rsi.applyOptions({ width: rsiRef.current.clientWidth });
    });
    ro.observe(candleRef.current);
    return () => {
      ro.disconnect();
      main.remove();
      rsi.remove();
    };
  }, [chart, signal]);

  return (
    <div>
      <div ref={candleRef} className="w-full" />
      <div className="text-[10px] text-mute px-2 py-1 border-y border-line">RSI(14)</div>
      <div ref={rsiRef} className="w-full" />
      <div className="flex gap-3 px-3 py-2 text-[10px] text-mute">
        <span className="text-[#4d9fff]">EMA9</span>
        <span className="text-[#e7c547]">EMA21</span>
        <span className="text-[#c084fc]">EMA50</span>
        <span className="text-[#fb923c]">VWAP</span>
        <span>Vol overlay</span>
      </div>
    </div>
  );
}
