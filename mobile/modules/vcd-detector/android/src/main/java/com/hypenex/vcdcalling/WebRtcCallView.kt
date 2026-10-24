// Vendored from DevanshModi09/LiveCalling (commit 1a01138, 2026-10-24), android/src/main/java/com/vcd/calling/WebRtcCallView.kt.
// Only the package name changed. Foreground WebView WebRTC call with a 16 kHz remote-PCM bridge.
package com.hypenex.vcdcalling

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.pm.PackageManager
import android.content.Intent
import android.net.Uri
import android.util.Base64
import android.webkit.JavascriptInterface
import android.webkit.PermissionRequest
import android.webkit.WebChromeClient
import android.webkit.WebResourceError
import android.webkit.WebResourceRequest
import android.webkit.WebView
import android.webkit.WebViewClient
import java.io.Closeable
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Foreground-only WebRTC view. Construct on the UI thread after microphone permission is granted. */
@SuppressLint("SetJavaScriptEnabled")
class WebRtcCallView(
    private val activity: Activity,
    serverUrl: String,
    /** Called on WebView's bridge thread. Enqueue immediately; never run inference here. */
    private val onRemotePcm16k: (FloatArray) -> Unit,
    /** Called on the main thread. */
    private val onState: (String) -> Unit,
) : Closeable {
    private val trusted = Uri.parse(serverUrl)
    @Volatile private var closed = false
    val view = WebView(activity)

    private fun sameOrigin(uri: Uri): Boolean = uri.scheme == trusted.scheme &&
        uri.host == trusted.host && (if (uri.port == -1) 443 else uri.port) ==
        (if (trusted.port == -1) 443 else trusted.port)

    init {
        require(trusted.scheme == "https" && !trusted.host.isNullOrEmpty() && trusted.userInfo == null) {
            "Use your trusted HTTPS calling server URL"
        }
        require(activity.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) {
            "Request RECORD_AUDIO before creating the call view"
        }
        view.settings.apply {
            javaScriptEnabled = true
            domStorageEnabled = true
            mediaPlaybackRequiresUserGesture = false
            allowFileAccess = false
            allowContentAccess = false
            mixedContentMode = android.webkit.WebSettings.MIXED_CONTENT_NEVER_ALLOW
        }
        view.webViewClient = object : WebViewClient() {
            override fun onReceivedError(v: WebView, request: WebResourceRequest, error: WebResourceError) {
                if (request.isForMainFrame && !closed) onState("Could not load calling server. Check the address and connection.")
            }
            override fun shouldOverrideUrlLoading(v: WebView, request: WebResourceRequest): Boolean =
                !sameOrigin(request.url)
        }
        view.webChromeClient = object : WebChromeClient() {
            override fun onPermissionRequest(request: PermissionRequest) {
                activity.runOnUiThread {
                    if (!closed && sameOrigin(request.origin) &&
                        activity.checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED &&
                        PermissionRequest.RESOURCE_AUDIO_CAPTURE in request.resources) {
                        request.grant(arrayOf(PermissionRequest.RESOURCE_AUDIO_CAPTURE))
                    } else request.deny()
                }
            }
        }
        view.addJavascriptInterface(Bridge(), "VoiceCheckAndroid")
        view.loadUrl(serverUrl)
    }

    private inner class Bridge {
        @JavascriptInterface fun shareRoom(room: String) {
            if (closed || !Regex("^[a-zA-Z0-9_-]{8,64}$").matches(room)) return
            val invitation = trusted.buildUpon().path("/").clearQuery().fragment(null)
                .appendQueryParameter("room", room).build().toString()
            activity.runOnUiThread {
                if (!closed) activity.startActivity(Intent.createChooser(
                    Intent(Intent.ACTION_SEND).setType("text/plain")
                        .putExtra(Intent.EXTRA_TEXT, "Join my Voice Check audio call: $invitation"),
                    "Share room link",
                ))
            }
        }

        @JavascriptInterface fun onCallState(state: String) {
            if (!closed) activity.runOnUiThread { if (!closed) onState(state.take(300)) }
        }
        @JavascriptInterface fun onRemotePcm(encoded: String) {
            if (closed || encoded.length != 856) return // exactly 320 mono PCM16 samples
            val bytes = try { Base64.decode(encoded, Base64.NO_WRAP) } catch (_: IllegalArgumentException) { return }
            if (bytes.size != 640) return
            val buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            val samples = FloatArray(320) { buffer.short / 32768f }
            if (!closed) onRemotePcm16k(samples)
        }
    }

    /** Call on UI thread from the screen/Activity destruction path. Stops microphone and peer connection. */
    override fun close() {
        if (closed) return
        closed = true
        view.evaluateJavascript("window.voiceCheckHangup && window.voiceCheckHangup()", null)
        view.removeJavascriptInterface("VoiceCheckAndroid")
        view.stopLoading()
        view.loadUrl("about:blank")
        view.destroy()
    }
}
