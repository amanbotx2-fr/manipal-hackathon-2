#!/usr/bin/env bash
# One-shot setup of the Voice Check calling server + TURN on an Ubuntu Lightsail instance.
# Run as root (sudo -i) on the instance:  bash setup-lightsail.sh <hostname> [turn-password]
#   <hostname>      a DNS name pointing at this instance's STATIC IP, e.g. call.example.com,
#                   or, with no domain, the free wildcard DNS form  <static-ip-with-dashes>.sslip.io
#                   (e.g. 13-233-1-2.sslip.io). Caddy obtains a Let's Encrypt certificate for it.
#   [turn-password] optional; generated if omitted. Printed at the end.
# Before running: in the Lightsail console → instance → Networking, add firewall rules
#   TCP 80, TCP 443, TCP 3478, UDP 3478, UDP 49160-49200   and attach a static IP.
set -euo pipefail
HOST="${1:?hostname required (e.g. 13-233-1-2.sslip.io)}"
TURN_PASS="${2:-$(head -c 24 /dev/urandom | base64 | tr -d '/+=' | head -c 24)}"
REPO="https://github.com/amanbotx2-fr/manipal-hackathon-2.git"
BRANCH="live-calling"
PUBLIC_IP="$(curl -fsS https://checkip.amazonaws.com || curl -fsS https://api.ipify.org)"
echo "== host $HOST  public ip $PUBLIC_IP"

# --- packages: Node 20, Caddy, coturn ---------------------------------------------------------
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y curl git ca-certificates coturn debian-keyring debian-archive-keyring apt-transport-https gnupg
if ! command -v node >/dev/null || [ "$(node -e 'console.log(process.versions.node.split(".")[0])')" -lt 20 ]; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
if ! command -v caddy >/dev/null; then
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' > /etc/apt/sources.list.d/caddy-stable.list
  apt-get update -y && apt-get install -y caddy
fi

# --- app code (calling/ folder of the repo, live-calling branch) ------------------------------
id -u livecalling >/dev/null 2>&1 || useradd --system --home /opt/LiveCalling --shell /usr/sbin/nologin livecalling
rm -rf /tmp/vcd-src && git clone --depth 1 --branch "$BRANCH" "$REPO" /tmp/vcd-src
mkdir -p /opt/LiveCalling && rsync -a --delete /tmp/vcd-src/calling/ /opt/LiveCalling/ 2>/dev/null || { rm -rf /opt/LiveCalling && cp -R /tmp/vcd-src/calling /opt/LiveCalling; }
(cd /opt/LiveCalling/server && npm ci --omit=dev --no-audit --no-fund)
chown -R livecalling:livecalling /opt/LiveCalling

# --- coturn -----------------------------------------------------------------------------------
cat > /etc/turnserver.conf <<EOF
listening-port=3478
fingerprint
lt-cred-mech
user=voicecheck:${TURN_PASS}
realm=${HOST}
external-ip=${PUBLIC_IP}
min-port=49160
max-port=49200
no-multicast-peers
no-cli
EOF
sed -i 's/^#\?TURNSERVER_ENABLED=.*/TURNSERVER_ENABLED=1/' /etc/default/coturn 2>/dev/null || echo 'TURNSERVER_ENABLED=1' >> /etc/default/coturn
systemctl enable --now coturn && systemctl restart coturn

# --- signaling server (systemd) ---------------------------------------------------------------
cat > /etc/livecalling.env <<EOF
PORT=8788
ALLOWED_ORIGIN=https://${HOST}
TURN_URLS=turn:${HOST}:3478?transport=udp,turn:${HOST}:3478?transport=tcp
TURN_USERNAME=voicecheck
TURN_CREDENTIAL=${TURN_PASS}
EOF
chmod 600 /etc/livecalling.env
cp /opt/LiveCalling/deploy/livecalling.service /etc/systemd/system/livecalling.service
systemctl daemon-reload && systemctl enable --now livecalling && systemctl restart livecalling

# --- HTTPS reverse proxy ----------------------------------------------------------------------
cat > /etc/caddy/Caddyfile <<EOF
${HOST} {
    reverse_proxy 127.0.0.1:8788
}
EOF
systemctl enable --now caddy && systemctl reload caddy || systemctl restart caddy

sleep 3
echo
echo "== status"; systemctl is-active livecalling coturn caddy | paste -sd' ' -
echo "== /config (via local port):"; curl -fsS http://127.0.0.1:8788/config; echo
echo
echo "Calling page:   https://${HOST}/"
echo "TURN user/pass: voicecheck / ${TURN_PASS}   (also in /etc/livecalling.env)"
echo "If https://${HOST}/ does not load within a minute, check: journalctl -u caddy -n 30 (DNS/port 80 for the certificate)."
