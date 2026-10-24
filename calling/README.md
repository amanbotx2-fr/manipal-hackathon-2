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
