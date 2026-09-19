import "./globals.css";
import type { Metadata } from "next";
import { Shell } from "@/components/Shell";

export const metadata: Metadata = {
  title: "Yami Financier",
  description: "Decision-support scanner for Binance USDT spot crypto. Not a broker. Not advice.",
  applicationName: "Yami Financier",
  icons: {
    icon: [
      { url: "/favicon.ico" },
      { url: "/yf-mark.svg", type: "image/svg+xml" },
      { url: "/favicon-32.png", sizes: "32x32", type: "image/png" },
    ],
    apple: [{ url: "/yf-icon.png" }],
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="font-sans antialiased">
        <Shell>{children}</Shell>
      </body>
    </html>
  );
}
