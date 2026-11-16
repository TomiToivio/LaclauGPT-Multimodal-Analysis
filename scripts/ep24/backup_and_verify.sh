#!/usr/bin/env bash
# #373: immutable, separately stored snapshot with actual restore/hash validation.
set -euo pipefail
usage() {
  echo 'Usage: bash scripts/ep24/backup_and_verify.sh LOCAL_PRIVATE_OUTPUT_ROOT RCLONE_PRIVATE_REMOTE_PREFIX' >&2
  echo 'Example remote: allas-private:ep24-backups (configured using private rclone credentials)' >&2
}
[[ $# -eq 2 ]] || { usage; exit 2; }
SOURCE="$(realpath "$1")"
REMOTE="${2%/}"
[[ -d "$SOURCE" && -n "$REMOTE" && "$REMOTE" == *:* ]] || { usage; exit 2; }
command -v rclone >/dev/null || { echo "rclone is required" >&2; exit 2; }
command -v sha256sum >/dev/null || { echo "sha256sum is required" >&2; exit 2; }
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
VERSION="ep24_${STAMP}_$$"
TARGET="${REMOTE}/${VERSION}"
TMP="$(mktemp -d)"
chmod 700 "$TMP"
trap 'rm -rf "$TMP"' EXIT

# Verify every local file first, then copy to a new remote version (never overwrite last good).
python3 - "$SOURCE" "$TMP/checksums.sha256" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
manifest = pathlib.Path(sys.argv[2])
paths = sorted(p for p in root.rglob("*") if p.is_file())
if not paths:
    raise SystemExit("No files to back up")
with manifest.open("w", encoding="utf-8") as out:
    for path in paths:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        relative = path.relative_to(root)
        if "\n" in str(relative) or "\r" in str(relative):
            raise SystemExit("Unsafe filename")
        out.write(f"{h.hexdigest()}  {relative}\n")
print(f"Prepared {len(paths)} file hashes for independent storage")
PY
# Explicitly fail if remote destination already exists; never coalesce two snapshots.
if rclone lsf "$REMOTE" --dirs-only 2>/dev/null | grep -Fxq "${VERSION}/"; then
  echo "Remote snapshot destination already exists: refuse overwrite" >&2
  exit 2
fi
rclone copy "$SOURCE" "$TARGET" --checkers 4 --transfers 4
rclone copyto "$TMP/checksums.sha256" "$TARGET/checksums.sha256"
rclone copy "$TARGET" "$TMP/restored" --checkers 4 --transfers 4
(cd "$TMP/restored" && sha256sum -c checksums.sha256)
echo "Off-node snapshot and clean restore verified: $TARGET"
echo "Store the remote version and restore proof in the private run log."
