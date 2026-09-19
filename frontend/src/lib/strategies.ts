/** Short hover blurbs for scanner/dashboard strategy labels. Max two sentences. */
export const STRATEGY_DESCRIPTIONS: Record<string, string> = {
  volume_confirmed_breakout:
    "Longs a close through 20-bar resistance backed by expanded volume. Watches when volume is there but the close-through is not confirmed.",
  trend_pullback:
    "Longs a dip into the 21 EMA in an uptrend that then reclaims it. Watches while price is still sitting on the average.",
  momentum_continuation:
    "Longs a stacked-EMA uptrend with higher closes, volume, and price above VWAP. Watches when the stack is there but follow-through is incomplete.",
  failed_breakout:
    "Shorts a poke through resistance that fails and closes back below, with a rejection wick. Watches when the reject is missing volume or wick confirmation.",
  parabolic_exhaustion:
    "Shorts an overextended rally as RSI rolls over from overbought and climax volume fades. Watches when price is stretched but has not started to roll over.",
  breakdown_continuation:
    "Shorts a stacked-down trend that closes through 20-bar support on volume. Watches when the stack is into support without a full breakdown.",
};

export function strategyDescription(id?: string | null): string | null {
  if (!id) return null;
  return STRATEGY_DESCRIPTIONS[id] || null;
}
