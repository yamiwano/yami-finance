const { app, BrowserWindow } = require("electron");
const { spawn, spawnSync } = require("child_process");
const fs = require("fs");
const http = require("http");
const path = require("path");

const ROOT = path.resolve(__dirname, "..");
const BACKEND = path.join(ROOT, "backend");
const FRONTEND = path.join(ROOT, "frontend");
const ICON = path.join(__dirname, "icons", "icon.png");

const API_PORT = Number(process.env.YAMI_API_PORT || 18765);
const UI_PORT = Number(process.env.YAMI_UI_PORT || 18766);
const API_BASE = `http://127.0.0.1:${API_PORT}`;
const UI_BASE = `http://127.0.0.1:${UI_PORT}`;
const WS_BASE = `ws://127.0.0.1:${API_PORT}/ws`;

let backendProc = null;
let frontendProc = null;
let mainWindow = null;
let shuttingDown = false;

app.setName("Yami Financier");
app.setPath("userData", path.join(app.getPath("appData"), "YamiFinancier"));

function pythonBin() {
  const candidates = [
    path.join(BACKEND, ".venv312", "bin", "python"),
    path.join(BACKEND, ".venv", "bin", "python"),
    "python3",
  ];
  return candidates.find((c) => c === "python3" || fs.existsSync(c));
}

function spawnTracked(command, args, opts) {
  const child = spawn(command, args, {
    ...opts,
    detached: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  child.stdout?.on("data", (d) => process.stdout.write(d));
  child.stderr?.on("data", (d) => process.stderr.write(d));
  child.unref();
  return child;
}

function stopChild(child) {
  if (!child?.pid) return;
  try {
    process.kill(-child.pid, "SIGTERM");
  } catch {
    try {
      child.kill("SIGTERM");
    } catch {
      /* already gone */
    }
  }
}

function shutdown() {
  if (shuttingDown) return;
  shuttingDown = true;
  const front = frontendProc;
  const back = backendProc;
  frontendProc = null;
  backendProc = null;
  stopChild(front);
  stopChild(back);
  setTimeout(() => {
    for (const child of [front, back]) {
      if (!child?.pid) continue;
      try {
        process.kill(-child.pid, "SIGKILL");
      } catch {
        /* ignore */
      }
    }
  }, 1500).unref();
}

function waitForHttp(url, timeoutMs = 60000) {
  const started = Date.now();
  return new Promise((resolve, reject) => {
    const tick = () => {
      const req = http.get(url, (res) => {
        res.resume();
        resolve();
      });
      req.on("error", () => {
        if (Date.now() - started > timeoutMs) {
          reject(new Error(`Timed out waiting for ${url}`));
          return;
        }
        setTimeout(tick, 250);
      });
    };
    tick();
  });
}

function hasProdBuild() {
  return fs.existsSync(path.join(FRONTEND, ".next", "BUILD_ID"));
}

function ensureFrontendBuild() {
  if (hasProdBuild()) return;
  const npm = process.platform === "win32" ? "npm.cmd" : "npm";
  const result = spawnSync(npm, ["run", "build"], { cwd: FRONTEND, stdio: "inherit", env: process.env });
  if (result.status !== 0) {
    throw new Error("Frontend production build failed");
  }
}

function startServices() {
  const py = pythonBin();
  backendProc = spawnTracked(
    py,
    ["-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", String(API_PORT)],
    {
      cwd: BACKEND,
      env: {
        ...process.env,
        PYTHONPATH: BACKEND,
        CORS_ORIGINS: UI_BASE,
      },
    }
  );

  const nextBin = path.join(FRONTEND, "node_modules", ".bin", "next");
  frontendProc = spawnTracked(nextBin, [hasProdBuild() ? "start" : "dev", "-H", "127.0.0.1", "-p", String(UI_PORT)], {
    cwd: FRONTEND,
    env: {
      ...process.env,
      PORT: String(UI_PORT),
      NEXT_PUBLIC_API_URL: API_BASE,
      NEXT_PUBLIC_WS_URL: WS_BASE,
    },
  });
}

function splashHtml() {
  return `data:text/html;charset=utf-8,${encodeURIComponent(`<!doctype html>
<html><head><title>Yami Financier</title>
<style>
  html,body{height:100%;margin:0;background:#07090f;color:#d5dce8;font-family:IBM Plex Sans,system-ui,sans-serif;display:flex;align-items:center;justify-content:center}
  .wrap{text-align:center}
  .yf{font-size:42px;font-weight:800;color:#4d9fff;letter-spacing:.04em}
  .bar{width:48px;height:3px;background:#e7c547;margin:10px auto 16px;border-radius:2px}
  p{color:#6b778c;font-size:13px}
</style></head>
<body><div class="wrap"><div class="yf">YF</div><div class="bar"></div><p>Starting Yami Financier…</p></div></body></html>`)}`;
}

async function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1440,
    height: 900,
    minWidth: 1100,
    minHeight: 720,
    title: "Yami Financier",
    icon: ICON,
    backgroundColor: "#07090f",
    autoHideMenuBar: true,
    show: false,
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      additionalArguments: [`--yami-api=${API_BASE}`, `--yami-ws=${WS_BASE}`],
      sandbox: false,
    },
  });
  mainWindow.setMenuBarVisibility(false);
  mainWindow.loadURL(splashHtml());
  mainWindow.once("ready-to-show", () => mainWindow?.show());

  ensureFrontendBuild();
  startServices();
  await waitForHttp(`${API_BASE}/api/health`);
  await waitForHttp(UI_BASE);
  if (!mainWindow || shuttingDown) return;
  await mainWindow.loadURL(UI_BASE);

  mainWindow.on("closed", () => {
    mainWindow = null;
    shutdown();
    app.quit();
  });
}

app.whenReady().then(createWindow).catch((err) => {
  console.error(err);
  shutdown();
  app.quit();
});

app.on("window-all-closed", () => {
  shutdown();
  app.quit();
});

app.on("before-quit", () => {
  shutdown();
});

process.on("SIGINT", () => {
  shutdown();
  app.quit();
});
process.on("SIGTERM", () => {
  shutdown();
  app.quit();
});
