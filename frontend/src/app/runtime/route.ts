import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export function GET() {
  const api = process.env.RADAR_API || process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const ws = api.replace(/^http/i, "ws").replace(/\/$/, "") + "/ws";
  return NextResponse.json({ apiBase: api, wsBase: ws });
}
