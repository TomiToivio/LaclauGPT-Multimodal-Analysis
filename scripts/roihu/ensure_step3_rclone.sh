#!/usr/bin/env bash
# Install/resolve a persistent native ARM64 rclone for the Roihu GPU jobs.
# Intended to be sourced after the appropriate CSC modules are loaded.
roihu_ensure_rclone() {
  # The Allas module exposes the site's rclone binary and configured remotes.
  # Explicitly load it here, after any module purges/venv activation, so both
  # the generic and legacy Step 3 launchers inherit a usable command.
  if type module >/dev/null 2>&1; then
    module load allas || {
      echo "Step 3 requires 'module load allas' on CSC Roihu." >&2
      return 2
    }
  fi
  local root="${LACLAUGPT_MULTIMODAL_PRIVATE_ROOT:?private root required}"
  local bin_dir="${LACLAUGPT_RCLONE_BIN_DIR:-${root}/.tools/bin}"
  local destination="${bin_dir}/rclone"
  mkdir -p "${bin_dir}"
  if [[ -x "${destination}" ]] && "${destination}" version >/dev/null 2>&1; then
    export PATH="${bin_dir}:${PATH}"
    return 0
  fi
  if command -v rclone >/dev/null 2>&1 && rclone version >/dev/null 2>&1; then
    local existing
    existing="$(command -v rclone)"
    if [[ "${existing}" != "${destination}" ]]; then
      cp "${existing}" "${destination}"
      chmod 755 "${destination}"
    fi
  else
    echo "Roihu Step 3: rclone missing; installing native ARM64 rclone to private project storage." >&2
    local tmp
    tmp="$(mktemp -d)"
    if ! curl -fL --retry 3 https://downloads.rclone.org/rclone-current-linux-arm64.zip -o "${tmp}/rclone.zip"; then
      rm -rf "${tmp}"
      echo "rclone download failed. Run bash scripts/roihu/setup_step_3_video.sh on a network-enabled Roihu node." >&2
      return 2
    fi
    if ! python3 - "${tmp}/rclone.zip" "${destination}.tmp" <<'PY'
import pathlib, sys, zipfile
archive, target = map(pathlib.Path, sys.argv[1:])
with zipfile.ZipFile(archive) as zf:
    candidates = [info for info in zf.infolist()
                  if info.filename.endswith("/rclone") and not info.is_dir()]
    if len(candidates) != 1:
        raise ValueError("Expected one rclone executable in official archive")
    target.write_bytes(zf.read(candidates[0]))
PY
    then
      rm -rf "${tmp}" "${destination}.tmp"
      return 2
    fi
    chmod 755 "${destination}.tmp"
    mv -f "${destination}.tmp" "${destination}"
    rm -rf "${tmp}"
  fi
  export PATH="${bin_dir}:${PATH}"
  if ! rclone version >/dev/null 2>&1; then
    echo "Installed rclone cannot run on $(uname -m); rerun Step 3 setup on ARM64." >&2
    return 2
  fi
  echo "Roihu Step 3 rclone ready: $(command -v rclone)" >&2
}
