const { contextBridge } = require("electron");

function arg(name) {
  const prefix = `--${name}=`;
  const hit = process.argv.find((a) => a.startsWith(prefix));
  return hit ? hit.slice(prefix.length) : "";
}

const apiBase = arg("yami-api") || process.env.YAMI_API_BASE || "http://127.0.0.1:18765";
const wsBase = arg("yami-ws") || process.env.YAMI_WS_BASE || "ws://127.0.0.1:18765/ws";

contextBridge.exposeInMainWorld("yamiDesktop", { apiBase, wsBase });
