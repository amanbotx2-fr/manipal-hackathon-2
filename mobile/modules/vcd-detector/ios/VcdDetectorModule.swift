import ExpoModulesCore

// iOS on-device inference is NOT implemented yet (Android first). Same API surface as
// android/.../VcdDetectorModule.kt so the JS loads; every inference call rejects clearly.
// Planned: onnxruntime-objc pod 1.31.0, AVAudioFile decode, same Preprocess/Resampler port.
public class VcdDetectorModule: Module {
  private let notImplemented = "On-device detection is not implemented on iOS yet"

  public func definition() -> ModuleDefinition {
    Name("VcdDetector")

    Events("onInstallProgress")

    AsyncFunction("getStatus") { () -> [String: Any?] in
      return ["installed": false, "installing": false, "loaded": false, "referenceAvailable": false,
              "modelBytes": 0, "partialBytes": 0, "expectedBytes": 1262760439,
              "sha256": "d1c232bf4d7990526804b375c15804c0ee4e9b566478b908c5a57431491b7342",
              "revision": "ee0187142973e1f02050e47d2a31c55ee0263065", "modelPath": "", "freeBytes": nil,
              "lastLoad": nil, "attribution": self.notImplemented]
    }
    AsyncFunction("getAdapterSpec") { () throws -> Bool in throw Exception(name: "ERR_IOS", description: self.notImplemented) }
    AsyncFunction("listAdapters") { () -> [Any] in return [] }
    AsyncFunction("installModel") { () throws -> Bool in throw Exception(name: "ERR_IOS", description: self.notImplemented) }
    AsyncFunction("cancelInstall") { () in }
    AsyncFunction("deleteModel") { () -> Bool in return true }
    AsyncFunction("loadModel") { () throws -> Bool in throw Exception(name: "ERR_IOS", description: self.notImplemented) }
    AsyncFunction("unloadModel") { () -> Bool in return true }
    AsyncFunction("analyseFile") { (_: String, _: String, _: Bool, _: Bool) throws -> Bool in
      throw Exception(name: "ERR_IOS", description: self.notImplemented)
    }
    AsyncFunction("analysePcm") { (_: Data, _: Int, _: Int, _: String) throws -> Bool in
      throw Exception(name: "ERR_IOS", description: self.notImplemented)
    }
    AsyncFunction("runReferenceTensors") { (_: Int) throws -> Bool in
      throw Exception(name: "ERR_IOS", description: self.notImplemented)
    }
    AsyncFunction("getMemory") { () -> [String: Any?] in return [:] }
  }
}
