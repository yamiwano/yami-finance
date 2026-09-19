import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#07090f",
        panel: "#0e121b",
        panel2: "#121826",
        line: "#1c2433",
        ink: "#d5dce8",
        mute: "#6b778c",
        long: "#3dd68c",
        short: "#f0616d",
        watch: "#e7c547",
        avoid: "#8b93a7",
        accent: "#4d9fff",
      },
      fontFamily: {
        sans: ["IBM Plex Sans", "system-ui", "sans-serif"],
        mono: ["IBM Plex Mono", "ui-monospace", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
