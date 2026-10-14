#!/usr/bin/env bash
# Update AUR packages after a GitHub release.
# Usage: ./scripts/aur-publish.sh 0.8.1
#
# Prerequisites:
#   - SSH key added to AUR profile
#   - GitHub tag v$VERSION pushed (tarball must exist)
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.8.1"
    exit 1
fi

VERSION="$1"
TARBALL_URL="https://github.com/JaimeAlonsoGA/simic-monorepo/archive/v$VERSION.tar.gz"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AUR_DIR="/tmp/aur-simic-release"

# Verify tarball exists
echo "Checking tarball at $TARBALL_URL..."
HTTP_CODE=$(curl -sL -o /dev/null -w '%{http_code}' "$TARBALL_URL")
if [ "$HTTP_CODE" != "200" ]; then
    echo "Error: tarball returned HTTP $HTTP_CODE. Did you push the v$VERSION tag?"
    exit 1
fi

# Compute sha256sum
echo "Computing sha256sum..."
SHA256=$(curl -sL "$TARBALL_URL" | sha256sum | cut -d' ' -f1)
echo "  $SHA256"

rm -rf "$AUR_DIR"
mkdir -p "$AUR_DIR"

# --- simic (main package) ---
echo ""
echo "==> Updating simic..."
git clone ssh://aur@aur.archlinux.org/simic.git "$AUR_DIR/simic"
cp "$REPO_ROOT/packaging/PKGBUILD" "$AUR_DIR/simic/PKGBUILD"
sed -i "s/sha256sums=('SKIP')/sha256sums=('$SHA256')/" "$AUR_DIR/simic/PKGBUILD"
(cd "$AUR_DIR/simic" && makepkg --printsrcinfo > .SRCINFO)
(cd "$AUR_DIR/simic" && git add PKGBUILD .SRCINFO && git commit -m "Update to $VERSION" && git push)

# --- simic-llm ---
echo ""
echo "==> Updating simic-llm..."
git clone ssh://aur@aur.archlinux.org/simic-llm.git "$AUR_DIR/simic-llm"
cp "$REPO_ROOT/packaging/simic-llm/PKGBUILD" "$AUR_DIR/simic-llm/PKGBUILD"
cp "$REPO_ROOT/packaging/simic-llm/simic-llm.install" "$AUR_DIR/simic-llm/"
(cd "$AUR_DIR/simic-llm" && makepkg --printsrcinfo > .SRCINFO)
(cd "$AUR_DIR/simic-llm" && git add PKGBUILD .SRCINFO simic-llm.install && git commit -m "Update to $VERSION" && git push)

# --- simic-diarize ---
echo ""
echo "==> Updating simic-diarize..."
git clone ssh://aur@aur.archlinux.org/simic-diarize.git "$AUR_DIR/simic-diarize"
cp "$REPO_ROOT/packaging/simic-diarize/PKGBUILD" "$AUR_DIR/simic-diarize/PKGBUILD"
cp "$REPO_ROOT/packaging/simic-diarize/simic-diarize.install" "$AUR_DIR/simic-diarize/"
(cd "$AUR_DIR/simic-diarize" && makepkg --printsrcinfo > .SRCINFO)
(cd "$AUR_DIR/simic-diarize" && git add PKGBUILD .SRCINFO simic-diarize.install && git commit -m "Update to $VERSION" && git push)

# --- simic-tray ---
echo ""
echo "==> Updating simic-tray..."
git clone ssh://aur@aur.archlinux.org/simic-tray.git "$AUR_DIR/simic-tray"
cp "$REPO_ROOT/packaging/simic-tray/PKGBUILD" "$AUR_DIR/simic-tray/PKGBUILD"
cp "$REPO_ROOT/packaging/simic-tray/simic-tray.install" "$AUR_DIR/simic-tray/"
(cd "$AUR_DIR/simic-tray" && makepkg --printsrcinfo > .SRCINFO)
(cd "$AUR_DIR/simic-tray" && git add PKGBUILD .SRCINFO simic-tray.install && git commit -m "Update to $VERSION" && git push)

# Cleanup
rm -rf "$AUR_DIR"

echo ""
echo "All AUR packages updated to $VERSION."
