# Deploying the calling server on AWS Lightsail

One Ubuntu instance runs all three parts: the Node signaling server, Caddy (HTTPS) and coturn (TURN).
The smallest plans are enough (signaling is tiny; TURN relays audio at ~50 kbit/s per call).

## 1. Instance and network
1. Create an Ubuntu 22.04 or 24.04 instance. Attach a **static IP** (Networking tab). TURN must
   advertise a fixed public address.
2. In the instance's **Networking → IPv4 Firewall** add:
   - TCP 80, TCP 443 (Caddy / HTTPS)
   - TCP 3478, UDP 3478 (TURN)
   - UDP 49160–49200 (TURN relay range)
3. Hostname for HTTPS. Either point a domain/subdomain (A record) at the static IP, or, with no
   domain, use the free wildcard DNS form `<static-ip-with-dashes>.sslip.io`
   (static IP 13.233.1.2 → `13-233-1-2.sslip.io`). Caddy fetches a Let's Encrypt certificate either way.

## 2. Install
SSH in (Lightsail's browser SSH works), then:

```sh
sudo -i
curl -fsSL https://raw.githubusercontent.com/amanbotx2-fr/manipal-hackathon-2/live-calling/calling/deploy/setup-lightsail.sh -o setup-lightsail.sh
bash setup-lightsail.sh 13-233-1-2.sslip.io
```

The script installs Node 20, Caddy and coturn, pulls the `calling/` folder from the `live-calling`
branch into `/opt/LiveCalling`, writes `/etc/livecalling.env` (ALLOWED_ORIGIN + TURN credentials),
`/etc/turnserver.conf` and `/etc/caddy/Caddyfile`, enables the three services, and prints the
calling page URL and the TURN password.

## 3. Check
- `https://<host>/config` must list the STUN servers **and** your `turn:` URLs.
- Open `https://<host>/` on two phones (mobile data on at least one), create a room, share the
  link, both tap **Join call**. The status line should read "Call connected · via TURN relay" or
  "direct (NAT traversal)". If it says "Could not connect", TURN ports are blocked: re-check the
  Lightsail firewall rules above.
- In the Voice Check app, enter `https://<host>/` as the calling server.

## 4. Update later
```sh
sudo -i; cd /tmp && rm -rf vcd-src && git clone --depth 1 --branch live-calling https://github.com/amanbotx2-fr/manipal-hackathon-2.git vcd-src \
 && rsync -a --delete --exclude node_modules vcd-src/calling/ /opt/LiveCalling/ && (cd /opt/LiveCalling/server && npm ci --omit=dev) \
 && chown -R livecalling:livecalling /opt/LiveCalling && systemctl restart livecalling
```

Logs: `journalctl -u livecalling -f`, `journalctl -u coturn -f`, `journalctl -u caddy -f`.
Security notes: room codes are the only access control; the TURN password is a long-term credential,
so rotate it after the event (edit `/etc/turnserver.conf` and `/etc/livecalling.env`, restart both services).

## Doing the console steps with the AWS CLI (from any machine with `aws` configured)

Set your region once (the one your instance is in, e.g. `ap-south-1`), then:

```sh
export AWS_DEFAULT_REGION=ap-south-1
aws lightsail get-instances --query 'instances[].{name:name,state:state.name,ip:publicIpAddress,static:isStaticIp,os:blueprintName}' --output table
```

1. Static IP (free while attached). Replace `INSTANCE` with the name from the table:
```sh
aws lightsail allocate-static-ip --static-ip-name vcd-call-ip
aws lightsail attach-static-ip --static-ip-name vcd-call-ip --instance-name INSTANCE
aws lightsail get-static-ip --static-ip-name vcd-call-ip --query 'staticIp.ipAddress' --output text
```
2. Firewall rules (adds to the existing ones):
```sh
for rule in "80 80 tcp" "443 443 tcp" "3478 3478 tcp" "3478 3478 udp" "49160 49200 udp"; do set -- $rule
  aws lightsail open-instance-public-ports --instance-name INSTANCE --port-info fromPort=$1,toPort=$2,protocol=$3
done
aws lightsail get-instance-port-states --instance-name INSTANCE --output table
```
3. SSH key and login (Ubuntu images use the `ubuntu` user):
```sh
aws lightsail download-default-key-pair --query privateKeyBase64 --output text > lightsail.pem && chmod 600 lightsail.pem
ssh -i lightsail.pem ubuntu@STATIC_IP
```
4. On the instance, run the setup with the sslip.io hostname built from the static IP (dots → dashes):
```sh
sudo -i
curl -fsSL https://raw.githubusercontent.com/amanbotx2-fr/manipal-hackathon-2/live-calling/calling/deploy/setup-lightsail.sh -o setup-lightsail.sh
bash setup-lightsail.sh 13-233-1-2.sslip.io
```
5. Check from anywhere: `curl https://13-233-1-2.sslip.io/config` must list `stun:` and `turn:` entries.
