package com.hypenex.vcdcalling

import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

/** Expo module exposing the in-app call view. Lives in the vcd-detector Gradle module so the call shares the detector engine. */
class VcdCallingModule : Module() {
  override fun definition() = ModuleDefinition {
    Name("VcdCalling")

    View(VcdCallView::class) {
      Events("onCallState", "onLiveResult")

      Prop("serverUrl") { view: VcdCallView, url: String -> view.setServerUrl(url) }
      Prop("liveCheck") { view: VcdCallView, enabled: Boolean -> view.liveCheckEnabled = enabled }
      Prop("aiVoice") { view: VcdCallView, on: Boolean -> view.aiVoice = on }

      AsyncFunction("hangup") { view: VcdCallView -> view.hangup() }
      AsyncFunction("liveStats") { view: VcdCallView -> view.liveStats() }
    }
  }
}
