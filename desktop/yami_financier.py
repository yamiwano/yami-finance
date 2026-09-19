#!/usr/bin/env python3
"""Yami Financier desktop window. Closing the window stops the scanner and UI."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import gi

gi.require_version("Gtk", "3.0")
gi.require_version("Gdk", "3.0")
gi.require_version("WebKit2", "4.1")
from gi.repository import Gdk, GLib, Gtk, WebKit2  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
FRONTEND = ROOT / "frontend"
ICON = ROOT / "desktop" / "icons" / "icon.png"

API_PORT = int(os.environ.get("YAMI_API_PORT", "18765"))
UI_PORT = int(os.environ.get("YAMI_UI_PORT", "18766"))
API_BASE = f"http://127.0.0.1:{API_PORT}"
UI_BASE = f"http://127.0.0.1:{UI_PORT}"
WS_BASE = f"ws://127.0.0.1:{API_PORT}/ws"


def python_bin() -> str:
    for candidate in (
        BACKEND / ".venv312" / "bin" / "python",
        BACKEND / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def wait_http(url: str, timeout: float = 60.0) -> None:
    deadline = time.time() + timeout
    last_err = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.5) as resp:
                if 200 <= getattr(resp, "status", 200) < 500:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_err = exc
            time.sleep(0.25)
    raise RuntimeError(f"Timed out waiting for {url}: {last_err}")


class YamiFinancier:
    def __init__(self) -> None:
        self.procs: list[subprocess.Popen] = []
        self.alive = True
        GLib.set_prgname("yami-financier")
        Gdk.set_program_class("Yami Financier")

        self.window = Gtk.Window()
        self.window.set_title("Yami Financier")
        self.window.set_default_size(1440, 900)
        self.window.set_role("yami-financier")
        if ICON.exists():
            self.window.set_icon_from_file(str(ICON))
        self.window.connect("destroy", self._on_close)
        self.window.connect("delete-event", self._on_delete)

        manager = WebKit2.UserContentManager()
        boot = (
            "window.yamiDesktop = "
            f"{{ apiBase: '{API_BASE}', wsBase: '{WS_BASE}' }};"
        )
        try:
            manager.add_script(
                WebKit2.UserScript.new(
                    boot,
                    WebKit2.UserContentInjectedFrames.ALL_FRAMES,
                    WebKit2.UserScriptInjectionTime.START,
                    None,
                    None,
                )
            )
        except TypeError:
            manager.add_script(
                WebKit2.UserScript.new(
                    boot,
                    WebKit2.UserContentInjectedFrames.ALL_FRAMES,
                    WebKit2.UserScriptInjectionTime.START,
                    [],
                    [],
                )
            )
        self.view = WebKit2.WebView.new_with_user_content_manager(manager)
        settings = self.view.get_settings()
        settings.set_property("enable-developer-extras", True)
        self.window.add(self.view)
        self.window.show_all()
        self.view.load_html(
            """<!doctype html><html><body style="margin:0;background:#07090f;color:#4d9fff;
            font-family:sans-serif;display:flex;align-items:center;justify-content:center;height:100vh">
            <div style="text-align:center"><div style="font-size:42px;font-weight:800">YF</div>
            <div style="width:48px;height:3px;background:#e7c547;margin:12px auto"></div>
            <div style="color:#6b778c;font-size:13px">Starting Yami Financier…</div></div></body></html>""",
            "about:blank",
        )

    def start_services(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(BACKEND)
        env["CORS_ORIGINS"] = UI_BASE
        env["RADAR_API"] = API_BASE
        env["NEXT_PUBLIC_API_URL"] = API_BASE
        env["NEXT_PUBLIC_WS_URL"] = WS_BASE
        backend = subprocess.Popen(
            [python_bin(), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(API_PORT)],
            cwd=BACKEND,
            env=env,
            start_new_session=True,
        )
        self.procs.append(backend)

        env["YAMI_DIST"] = ".next-desktop"
        next_bin = FRONTEND / "node_modules" / ".bin" / "next"
        frontend = subprocess.Popen(
            [str(next_bin), "dev", "-H", "127.0.0.1", "-p", str(UI_PORT)],
            cwd=FRONTEND,
            env=env,
            start_new_session=True,
        )
        self.procs.append(frontend)

    def load_app(self) -> None:
        try:
            wait_http(f"{API_BASE}/api/health")
            wait_http(UI_BASE)
        except Exception:
            if self.alive:
                GLib.idle_add(self.window.set_title, "Yami Financier — failed to start")
            return
        if self.alive:
            GLib.idle_add(self.view.load_uri, UI_BASE)

    def stop(self) -> None:
        self.alive = False
        for proc in self.procs:
            if proc.poll() is not None:
                continue
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                continue
        deadline = time.time() + 2
        for proc in self.procs:
            while proc.poll() is None and time.time() < deadline:
                time.sleep(0.05)
            if proc.poll() is None:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.procs.clear()

    def _on_delete(self, *_args):
        self.stop()
        return False

    def _on_close(self, *_args):
        self.stop()
        Gtk.main_quit()


def main() -> int:
    if not (FRONTEND / "node_modules" / ".bin" / "next").exists():
        subprocess.check_call(["npm", "install"], cwd=FRONTEND)
    app = YamiFinancier()
    app.start_services()
    threading.Thread(target=app.load_app, daemon=True).start()
    try:
        Gtk.main()
    finally:
        app.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
