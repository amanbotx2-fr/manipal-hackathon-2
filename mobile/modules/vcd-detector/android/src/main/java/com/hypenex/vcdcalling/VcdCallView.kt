package com.hypenex.vcdcalling

import android.content.Context
import android.view.ViewGroup
import com.hypenex.vcddetector.DetectorEngine
import expo.modules.kotlin.AppContext
import expo.modules.kotlin.viewevent.EventDispatcher
import expo.modules.kotlin.views.ExpoView

/**
 * React Native view hosting the WebRTC call (WebView) and feeding the remote participant's PCM
 * into LiveAnalyser. Created when `serverUrl` is set (RECORD_AUDIO must already be granted; JS
 * requests it first). Closed when detached from the window.
 */
class VcdCallView(context: Context, appContext: AppContext) : ExpoView(context, appContext) {
  val onCallState by EventDispatcher()
  val onLiveResult by EventDispatcher()

  private var call: WebRtcCallView? = null
  private var analyser: LiveAnalyser? = null
  private var url: String? = null
  var liveCheckEnabled = true
    set(v) { field = v; if (!v) analyser?.let { it.close(); analyser = null } else if (call != null && analyser == null) startAnalyser() }

  init {
    keepScreenOn = true // the call is foreground-only; a sleeping screen would pause the WebView and drop it
  }

  fun setServerUrl(serverUrl: String) {
    if (serverUrl == url) return
    teardown()
    url = serverUrl
    val activity = appContext.currentActivity ?: run { onCallState(mapOf("state" to "No activity")); return }
    try {
      if (liveCheckEnabled) startAnalyser()
      val c = WebRtcCallView(
        activity = activity,
        serverUrl = serverUrl,
        onRemotePcm16k = { samples -> analyser?.push(samples) },
        onState = { state -> onCallState(mapOf("state" to state)) },
      )
      call = c
      addView(c.view, ViewGroup.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.MATCH_PARENT))
      onCallState(mapOf("state" to "Opening calling room"))
    } catch (e: Exception) {
      onCallState(mapOf("state" to "Cannot start call: ${e.message}"))
    }
  }

  private fun startAnalyser() {
    val engine = DetectorEngine.get(context.applicationContext)
    // EventDispatcher payloads must be non-null values; null fields (e.g. prediction on a rejected
    // window) are dropped and read as undefined in JS.
    analyser = LiveAnalyser(engine) { payload ->
      val clean = HashMap<String, Any>(); for ((k, v) in payload) if (v != null) clean[k] = v
      post {
        onLiveResult(clean)
        // Tell the call page what we concluded about the remote voice; the page relays it to the peer.
        val v = org.json.JSONObject().put("status", payload["status"]).put("prediction", payload["prediction"])
          .put("score", payload["synthetic_score"]).put("window", payload["window_index"]).put("reason", payload["reject_reason"] ?: payload["message"])
        call?.view?.evaluateJavascript("window.voiceCheckVerdict && window.voiceCheckVerdict($v)", null)
      }
    }
  }

  fun liveStats(): Map<String, Any?> = analyser?.stats() ?: mapOf("enabled" to false)

  fun hangup() = teardown()

  private fun teardown() {
    analyser?.close(); analyser = null
    call?.let { c -> removeView(c.view); c.close() }
    call = null
    url = null
  }

  override fun onDetachedFromWindow() {
    teardown()
    super.onDetachedFromWindow()
  }
}
