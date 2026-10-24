package com.hypenex.vcddetector

import android.net.Uri
import expo.modules.kotlin.Promise
import expo.modules.kotlin.modules.Module
import expo.modules.kotlin.modules.ModuleDefinition

/**
 * JS bridge for on-device Dhwani inference. Heavy work runs on DetectorEngine's own threads;
 * these functions only dispatch and resolve. Audio never leaves the device.
 */
class VcdDetectorModule : Module() {
  private val engine by lazy { DetectorEngine(appContext.reactContext!!.applicationContext) }

  private fun Promise.fail(t: Throwable) {
    val code = when (t) {
      is OutOfMemoryError -> "ERR_OUT_OF_MEMORY"
      is IllegalStateException -> "ERR_STATE"
      is IllegalArgumentException -> "ERR_ARGUMENT"
      else -> "ERR_INFERENCE"
    }
    reject(code, "${t.javaClass.simpleName}: ${t.message}", t)
  }

  override fun definition() = ModuleDefinition {
    Name("VcdDetector")

    Events("onInstallProgress")

    AsyncFunction("getStatus") { engine.status() }

    // Declared contract of the active model adapter (version, checksum, rate, windows, labels, threshold).
    AsyncFunction("getAdapterSpec") { engine.adapter.spec.toMap() }

    AsyncFunction("listAdapters") { Adapters.list().map { it.toMap() } }

    AsyncFunction("installModel") {
      engine.startInstall { sendEvent("onInstallProgress", it) }
      true
    }

    AsyncFunction("cancelInstall") { engine.store.cancel() }

    AsyncFunction("deleteModel") { promise: Promise ->
      engine.submit({ engine.unload(); engine.store.delete(); true }, { promise.resolve(it) }, { promise.fail(it) })
    }

    AsyncFunction("loadModel") { promise: Promise ->
      engine.submit({ engine.load() }, { promise.resolve(it) }, { promise.fail(it) })
    }

    AsyncFunction("unloadModel") { promise: Promise ->
      engine.submit({ engine.unload(); true }, { promise.resolve(it) }, { promise.fail(it) })
    }

    AsyncFunction("analyseFile") { uri: String, mode: String, compare: Boolean, dump: Boolean, promise: Promise ->
      engine.submit({ engine.analyseUri(Uri.parse(uri), mode, compare, dump) }, { promise.resolve(it) }, { promise.fail(it) })
    }

    // Future in-app call audio: little-endian float32 PCM bytes, interleaved channels.
    AsyncFunction("analysePcm") { pcm: ByteArray, sampleRate: Int, channels: Int, mode: String, promise: Promise ->
      engine.submit({ engine.analysePcm(pcm, sampleRate, channels, mode) }, { promise.resolve(it) }, { promise.fail(it) })
    }

    AsyncFunction("runReferenceTensors") { maxWindows: Int, promise: Promise ->
      engine.submit({ engine.runReferenceTensors(maxWindows) }, { promise.resolve(it) }, { promise.fail(it) })
    }

    AsyncFunction("getMemory") { engine.memory() }
  }
}
