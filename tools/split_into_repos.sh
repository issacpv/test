#!/usr/bin/env bash
# Turn each projects/<slug>/ folder into its own standalone git repository (and
# optionally a GitHub repo). Run this on your laptop after cloning this monorepo.
#
# Usage:
#   tools/split_into_repos.sh                 # local repos only, under ../bme-projects/
#   tools/split_into_repos.sh --github USER   # also create GitHub repos with `gh` (private)
#   DEST=/path/to/dir tools/split_into_repos.sh
#
# Requirements: git; for --github: the GitHub CLI (`gh auth login` first).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${DEST:-$HERE/../bme-projects}"
GITHUB_USER=""
VISIBILITY="${VISIBILITY:-private}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --github) GITHUB_USER="$2"; shift 2 ;;
    --public) VISIBILITY="public"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$DEST"

for proj in "$HERE"/projects/*/; do
  slug="$(basename "$proj")"
  target="$DEST/$slug"
  if [[ -d "$target/.git" ]]; then
    echo "== $slug: already a repo, syncing files"
    rsync -a --exclude .git --exclude data/ --exclude outputs/ "$proj" "$target/"
  else
    echo "== $slug: creating repo at $target"
    mkdir -p "$target"
    rsync -a --exclude data/ --exclude outputs/ "$proj" "$target/"
    # keep data/README.md but not data contents
    mkdir -p "$target/data"
    [[ -f "$proj/data/README.md" ]] && cp "$proj/data/README.md" "$target/data/README.md"
    git -C "$target" init -q -b main
  fi
  git -C "$target" add -A
  if ! git -C "$target" diff --cached --quiet; then
    git -C "$target" commit -q -m "Project scaffold: $slug" || true
  fi
  if [[ -n "$GITHUB_USER" ]]; then
    if ! git -C "$target" remote get-url origin >/dev/null 2>&1; then
      gh repo create "$GITHUB_USER/$slug" --"$VISIBILITY" --source "$target" --remote origin --push \
        || echo "   (gh repo create failed for $slug; create it manually)"
    else
      git -C "$target" push -u origin main || true
    fi
  fi
done

echo
echo "Done. Repos are under: $DEST"
