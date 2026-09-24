#!/bin/bash
# Install / update the anya Pi service.  Run from a checkout of this repo:
#     sudo pi/install.sh
# Safe to re-run: it updates the code and venv and keeps config, queue and reels.
set -euo pipefail

[ "$(id -u)" = 0 ] || { echo "run with sudo"; exit 1; }
REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP=/opt/anya
DATA=/srv/anya
U=anya

echo "== packages"
apt-get update -q
apt-get install -y -q ffmpeg python3-venv python3-dev rsync samba exfatprogs

echo "== user $U"
id -u $U >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin $U
usermod -aG video,render $U     # the HEVC decoder's device nodes

echo "== code -> $APP/src"
mkdir -p $APP/src
# Only what the Pi runs: the pipeline, the walking model it imports, this dir.
rsync -a --delete --exclude '__pycache__' --exclude 'models/pose/' \
  "$REPO/pipeline" "$REPO/walking" "$REPO/pi" $APP/src/
chown -R $U:$U $APP

echo "== venv"
[ -x $APP/venv/bin/python ] || runuser -u $U -- python3 -m venv $APP/venv
runuser -u $U -- $APP/venv/bin/pip install -q --upgrade pip
runuser -u $U -- $APP/venv/bin/pip install -q -r $APP/src/pi/requirements-pi.txt

echo "== data dirs under $DATA"
mkdir -p $DATA/{inbox,work,reels,state/jobs,models,site}
[ -f $DATA/config.toml ] || cp $APP/src/pi/config.example.toml $DATA/config.toml
chown -R $U:$U $DATA
chmod 700 $DATA/state

echo "== swap (4 GB Pi: torch + a 4K decode can spike past RAM)"
if [ -f /etc/dphys-swapfile ]; then
  sed -i 's/^#\?CONF_SWAPSIZE=.*/CONF_SWAPSIZE=4096/' /etc/dphys-swapfile
  sed -i 's/^#\?CONF_MAXSWAP=.*/CONF_MAXSWAP=4096/' /etc/dphys-swapfile
  systemctl restart dphys-swapfile || true
else
  echo "   no dphys-swapfile here; make sure you have >= 4 GB swap or zram"
fi

echo "== samba share [anya-reels] -> $DATA/reels"
if ! grep -q '^\[anya-reels\]' /etc/samba/smb.conf; then
  cat >> /etc/samba/smb.conf <<EOS

[anya-reels]
   path = $DATA/reels
   valid users = $U
   read only = no
   force user = $U
EOS
  systemctl restart smbd
  echo "   set the share password with:  sudo smbpasswd -a $U"
fi

echo "== udev + systemd"
install -m 644 $APP/src/pi/udev/99-anya-camera.rules /etc/udev/rules.d/
install -m 644 $APP/src/pi/systemd/anya-ingest@.service /etc/systemd/system/
install -m 644 $APP/src/pi/systemd/anya-worker.service /etc/systemd/system/
chmod +x $APP/src/pi/bin/anya-ingest-device
udevadm control --reload
systemctl daemon-reload
systemctl enable anya-worker.service

echo "== NCNN pose export for the near pass (the far one is made per site on first use)"
runuser -u $U -- env PYTHONPATH=$APP/src ANYA_POSE_MODELS=$DATA/models \
  $APP/venv/bin/python -m pipeline.anya2.export_pose export --backend ncnn

systemctl restart anya-worker.service
echo
echo "Installed.  Next steps (pi/README.md):"
echo "  1. put a site profile in $DATA/site   (python -m pipeline.anya2.site save ...)"
echo "  2. sudo smbpasswd -a $U               (to reach the reels share)"
echo "  3. plug the camera in; follow with:  journalctl -fu anya-worker -u 'anya-ingest@*'"
