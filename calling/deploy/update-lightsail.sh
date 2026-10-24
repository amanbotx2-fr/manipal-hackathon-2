#!/usr/bin/env bash
# Update an existing Lightsail deployment to the latest calling/ code on the live-calling branch and
# publish the Android App Links statement (so https://<host>/?room=... opens the Voice Check app).
# Run as root on the instance:  sudo -i; bash update-lightsail.sh
set -euo pipefail
REPO="https://github.com/amanbotx2-fr/manipal-hackathon-2.git"; BRANCH="live-calling"
rm -rf /tmp/vcd-src && git clone -q --depth 1 --branch "$BRANCH" "$REPO" /tmp/vcd-src
rsync -a --delete --exclude node_modules /tmp/vcd-src/calling/ /opt/LiveCalling/
(cd /opt/LiveCalling/server && npm ci --omit=dev --no-audit --no-fund >/dev/null)
chown -R livecalling:livecalling /opt/LiveCalling
# App Links statement from deploy/assetlinks.json (single line in the env file)
if ! grep -q '^ASSETLINKS_JSON=' /etc/livecalling.env; then
  printf "ASSETLINKS_JSON='%s'\n" "$(tr -d '\n ' < /opt/LiveCalling/deploy/assetlinks.json)" >> /etc/livecalling.env
  echo "added ASSETLINKS_JSON to /etc/livecalling.env"
fi
systemctl restart livecalling
sleep 2
HOST=$(grep '^ALLOWED_ORIGIN=' /etc/livecalling.env | cut -d= -f2- | sed 's|https://||')
echo "== services: $(systemctl is-active livecalling coturn caddy | paste -sd' ' -)"
echo "== assetlinks: $(curl -fsS "https://${HOST}/.well-known/assetlinks.json" | head -c 120)..."
echo "== autojoin in page: $(curl -fsS "https://${HOST}/call.js" | grep -c autojoin) (expect >=1)"
echo "Done. Install the new APK on the phones AFTER this (Android verifies the link at install time)."
