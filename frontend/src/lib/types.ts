export type Direction = "LONG" | "SHORT" | "WATCH" | "AVOID";
export type AssetType = "crypto" | "stock";

export type ScoreComponents = {
  technical_structure: number;
  volume_activity: number;
  market_context: number;
  setup_quality: number;
  catalyst_context: number;
  penalties: number;
  penalty_reasons: string[];
  total: number;
  raw_total?: number;
  learner_multiplier?: number;
  learner_min_score_bump?: number;
  learner_weights?: Record<string, number>;
  learner_note?: string;
  learner_stance?: string;
  learner_tweaks?: {
    key: string;
    label: string;
    prior: number | boolean;
    live: number | boolean;
    prior_text?: string;
    live_text?: string;
    why: string;
  }[];
};

export type RiskFlag = { code: string; severity: string; detail: string };

export type Signal = {
  id: string;
  asset_id: string;
  symbol: string;
  asset_type: AssetType;
  direction: Direction;
  strategy: string;
  strategy_label?: string | null;
  timeframe: string;
  status: string;
  detected_at: string;
  closed_at?: string | null;
  score: number;
  current_price: number;
  entry_low: number;
  entry_high: number;
  invalidation: number;
  stop: number;
  target_1: number;
  target_2: number;
  risk_reward: number;
  r_multiple?: number | null;
  exit_price?: number | null;
  score_components: ScoreComponents;
  risk_flags: RiskFlag[];
  reasons: string[];
  evidence: Record<string, unknown>;
  snapshot: Record<string, unknown>;
  ai_explanation?: {
    thesis: string;
    evidence: string[];
    bear_case: string;
    what_to_watch: string;
    summary: string;
    provider?: string;
  } | null;
};

export type ScannerRow = {
  asset_id: string;
  symbol: string;
  name: string;
  asset_type: AssetType;
  exchange: string;
  price: number | null;
  change_pct: number | null;
  volume: number | null;
  spread_bps: number | null;
  bid?: number | null;
  ask?: number | null;
  signal: Signal | null;
};

export type Candle = {
  ts: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

export type ChartPayload = {
  symbol: string;
  timeframe: string;
  candles: Candle[];
  ema9: (number | null)[];
  ema21: (number | null)[];
  ema50: (number | null)[];
  vwap: (number | null)[];
  rsi: (number | null)[];
};

export type WatchItem = {
  id: string;
  asset_id: string;
  symbol: string;
  name: string;
  asset_type: AssetType;
  notes: string;
  created_at: string;
  price: number | null;
  change_pct: number | null;
};

export type Settings = {
  min_score: number;
  enable_crypto?: boolean;
  enable_stocks?: boolean;
  enabled_strategies: string[];
  enabled_timeframes: string[];
  learn_from_outcomes?: boolean;
};

export type TapeStatus = {
  provider?: string;
  source?: string;
  ready?: boolean;
  ws_connected?: boolean;
  universe?: number;
  quotes?: number;
  streams?: number;
  last_message_at?: string | null;
  stale_ms?: number | null;
};

export type LearnerTweak = {
  key: string;
  label: string;
  prior: number | boolean;
  live: number | boolean;
  prior_text?: string;
  live_text?: string;
  why: string;
};

export type LearnerChange = {
  kind: "knob" | "weight" | "stance" | "rank";
  strategy?: string;
  strategy_label?: string;
  key: string;
  label: string;
  prior: number | boolean | string;
  live: number | boolean | string;
  prior_text: string;
  live_text: string;
  why: string;
};

export type LearnerEvent = {
  ts: string;
  decided: number;
  origin?: string;
  summary: string;
  changes: LearnerChange[];
};

export type LearnerStrategyPulse = {
  id: string;
  label: string;
  stance?: string;
  stance_label?: string;
  n_changes: number;
};

export type LearnerStatusCode = "paused" | "waiting" | "watching" | "adjusting";

export type LearnerStatus = {
  enabled: boolean;
  decided: number;
  status: LearnerStatusCode;
  status_label: string;
  calibrated_at?: string | null;
  last_change_at: string | null;
  last_change_summary: string | null;
  tweaked_strategies: LearnerStrategyPulse[];
  last_changes?: LearnerChange[];
};

export type Health = {
  ok: boolean;
  scans: number;
  last_scan_at: string | null;
  provider: string;
  ready?: boolean;
  tape?: TapeStatus;
  learner?: LearnerStatus;
};

export type LearnerBucket = {
  raw_n: number;
  wins: number;
  losses: number;
  avg_r: number | null;
  win_rate: number | null;
  multiplier: number;
  min_score_bump: number;
  confidence: number;
  note: string;
  stance?: string;
  stance_label?: string;
};

export type LearnerProfile = LearnerStatus & {
  weighted_n: number;
  updated_at: string | null;
  weights: Record<string, number>;
  weight_deltas: Record<string, number>;
  prior_weights: Record<string, number>;
  weight_labels: Record<string, string>;
  strategy_labels: Record<string, string>;
  by_strategy: Record<string, LearnerBucket>;
  by_timeframe: Record<string, LearnerBucket>;
  by_direction: Record<string, LearnerBucket>;
  notes: string[];
  half_life_days: number;
  knobs?: Record<string, Record<string, number | boolean>>;
  strategy_tweaks?: Record<string, { stance: string; stance_label: string; changes: LearnerTweak[] }>;
  knob_labels?: Record<string, string>;
  stance_labels?: Record<string, string>;
  status_labels?: Record<string, string>;
  journal?: LearnerEvent[];
};

export type Performance = {
  total_signals: number;
  active: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  average_r: number;
  by_strategy: Record<string, { count: number; wins: number; losses: number; avg_r: number | null; win_rate: number | null }>;
  by_score_range: Record<string, { count: number; wins: number; losses: number; win_rate: number | null }>;
  by_direction: Record<string, { count: number; wins: number; losses: number; avg_r: number | null; win_rate: number | null }>;
  period_started_at?: string | null;
  learner?: LearnerProfile;
};

export type Activity = {
  ts: string;
  kind: string;
  message: string;
  symbol?: string;
  id?: string;
  changes?: LearnerChange[];
};
