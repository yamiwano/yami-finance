"use client";

export function BrandMark({ size = 32 }: { size?: number }) {
  return (
    <div
      className="shrink-0 rounded-[9px] overflow-hidden border border-line bg-bg"
      style={{ width: size, height: size }}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src="/yf-icon.png" alt="YF" width={size} height={size} className="block w-full h-full object-cover" />
    </div>
  );
}
