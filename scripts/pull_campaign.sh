#!/usr/bin/env bash
# Mirror a campaign directory from the cluster into this checkout, so the
# Streamlit app can render a run it did not start.
#
#   scripts/pull_campaign.sh cln025_slurm          # once
#   scripts/pull_campaign.sh cln025_slurm 60       # every 60s, until Ctrl-C
#
# The campaign database stores trajectory paths relative to the directory the
# campaign was launched from (`campaigns/<name>/rounds/round_001.dcd`), so a
# run started from the repository root on the cluster lands here already
# addressable — provided this mirror keeps the same relative layout, which is
# why the destination is not configurable.
#
# What is deliberately left behind: `cache/` (the serialized System, tens of
# MB, rebuilt on the cluster), `slurm/` (batch scripts and the checkpoint the
# jobs pass between themselves) and `*.chk`. Those are what a *resume* needs,
# and a resume belongs on the cluster where the campaign is running. Everything
# the viewer and the diagnostics read comes across.
set -euo pipefail

REMOTE="${MDPILOT_REMOTE:-cjmchoi@chestnut-login.seas.upenn.edu}"
REMOTE_DIR="${MDPILOT_REMOTE_DIR:-MDPilot}"

name="${1:-}"
interval="${2:-}"
if [[ -z "$name" ]]; then
    echo "usage: $0 <campaign-name> [interval-seconds]" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dest="$repo_root/campaigns"
mkdir -p "$dest"

# The campaign database is the one file being written *while* it is copied —
# the loop commits a row per round — and rsync of a live SQLite file can land a
# torn page. `Connection.backup` is safe against a concurrent writer, so the
# snapshot is taken on the remote and shipped as an ordinary file.
# A campaign that is still building its system has no database yet, and in a
# watch loop that is a thing to report rather than to stop for.
snapshot_db() {
    if ! ssh "$REMOTE" "python3 -c \"
import os, sqlite3, sys
db = '$REMOTE_DIR/campaigns/$name/state.db'
if not os.path.exists(db):
    sys.exit(3)
src = sqlite3.connect('file:' + db + '?mode=ro', uri=True)
dst = sqlite3.connect('/tmp/mdpilot-$name-state.db')
src.backup(dst); dst.close()
\"" 2>/dev/null; then
        echo "  (no state.db on the cluster yet — no round has been recorded)"
        return 0
    fi
    rsync -az "$REMOTE:/tmp/mdpilot-$name-state.db" "$dest/$name/state.db"
}

pull() {
    rsync -az --partial --info=stats1 \
        --exclude 'cache/' --exclude 'slurm/' --exclude '*.chk' \
        --exclude 'state.db' \
        "$REMOTE:$REMOTE_DIR/campaigns/$name/" "$dest/$name/"
    snapshot_db
}

if [[ -z "$interval" ]]; then
    pull
    echo "-> campaigns/$name"
    exit 0
fi

echo "mirroring campaigns/$name every ${interval}s (Ctrl-C to stop)"
while true; do
    pull
    sleep "$interval"
done
