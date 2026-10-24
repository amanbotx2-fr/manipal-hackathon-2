# calling/ — signaling server and web call page (vendored)

Vendored from Devansh Modi's **LiveCalling** kit
(https://github.com/DevanshModi09/LiveCalling, commit `1a01138`, 2026-10-24), unchanged:
`server/` (Node WebSocket signaling + static hosting, no audio relay), `web/` (browser WebRTC call
page with the 16 kHz remote-PCM worklet), `deploy/` (Caddy/systemd examples).

The Android side of the kit (`WebRtcCallView.kt`) lives in
`mobile/modules/vcd-detector/android/src/main/java/com/hypenex/vcdcalling/` next to our live analyser,
so the call shares the app's single loaded detector model.

Run locally: `cd calling/server && npm ci && npm start` (port 8788). The Android WebView requires an
**HTTPS** origin, so expose it through your own HTTPS reverse proxy or a temporary tunnel, and set
`ALLOWED_ORIGIN` to that exact origin. Cross-network calls need STUN/TURN (`ICE_SERVERS_JSON`).
Room codes are bearer invitations; there is no identity, push ringing or background-call service.

## Reliability: STUN/TURN (read this if calls have no audio or drop after ~10 s)

WebRTC needs ICE servers to connect phones that are on different networks. Without them the two
sides never find a route: nobody hears anything and the call fails about 10 s after joining.

- `server.mjs` now defaults to public STUN (`stun.l.google.com`) when started from the command line.
  STUN is enough when at least one side is on a normal home/office NAT.
- **Mobile data (carrier-grade NAT) on both sides usually needs TURN.** Run coturn next to the server
  (`deploy/turnserver.conf.example`) or use a TURN provider, then start the server with:

```sh
ALLOWED_ORIGIN=https://call.YOUR_DOMAIN.com TURN_URLS="turn:call.YOUR_DOMAIN.com:3478?transport=udp,turn:call.YOUR_DOMAIN.com:3478?transport=tcp" TURN_USERNAME=voicecheck TURN_CREDENTIAL='the password from turnserver.conf' npm start
```

`ICE_SERVERS_JSON` still overrides everything. The call page shows the route once connected
("direct (same network)", "direct (NAT traversal)" or "via TURN relay"), which tells you whether TURN
is actually being used.

Client changes in `web/call.js` (vs the vendored kit): a signaling-socket drop no longer ends a
connected call; `disconnected`/`failed` trigger an ICE restart from the initiator (up to 3 tries)
instead of an immediate hang-up; the failure message says what to configure.

Quick tunnels (`cloudflared tunnel --url …`) are fine for the signaling page in tests, but they add
latency and can drop; deploy the server on a real HTTPS host for the demo.
