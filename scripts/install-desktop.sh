#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/applications"
ICON_BASE="${XDG_DATA_HOME:-$HOME/.local/share}/icons/hicolor"

mkdir -p "$APP_DIR" \
  "$ICON_BASE/16x16/apps" \
  "$ICON_BASE/32x32/apps" \
  "$ICON_BASE/48x48/apps" \
  "$ICON_BASE/64x64/apps" \
  "$ICON_BASE/128x128/apps" \
  "$ICON_BASE/256x256/apps" \
  "$ICON_BASE/512x512/apps"

cp "$ROOT/desktop/icons/icon-16.png" "$ICON_BASE/16x16/apps/yami-financier.png"
cp "$ROOT/desktop/icons/icon-32.png" "$ICON_BASE/32x32/apps/yami-financier.png"
cp "$ROOT/desktop/icons/icon-48.png" "$ICON_BASE/48x48/apps/yami-financier.png"
cp "$ROOT/desktop/icons/icon-64.png" "$ICON_BASE/64x64/apps/yami-financier.png"
cp "$ROOT/desktop/icons/icon-128.png" "$ICON_BASE/128x128/apps/yami-financier.png"
cp "$ROOT/desktop/icons/icon-256.png" "$ICON_BASE/256x256/apps/yami-financier.png"
cp "$ROOT/desktop/icons/icon-512.png" "$ICON_BASE/512x512/apps/yami-financier.png"

cat > "$APP_DIR/yami-financier.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Yami Financier
GenericName=Market scanner
Comment=Decision-support scanner for crypto and U.S. penny names
Exec=$ROOT/scripts/yami-financier
Icon=yami-financier
Terminal=false
Categories=Finance;Office;
StartupNotify=true
StartupWMClass=Yami Financier
Keywords=yami;yf;finance;crypto;scanner;
EOF

chmod +x "$ROOT/scripts/yami-financier" "$APP_DIR/yami-financier.desktop"
command -v update-desktop-database >/dev/null && update-desktop-database "$APP_DIR" >/dev/null 2>&1 || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -f "$ICON_BASE" >/dev/null 2>&1 || true

echo "Installed Yami Financier launcher to $APP_DIR/yami-financier.desktop"
echo "Search the app menu for “Yami Financier”. Closing the window stops the app."
