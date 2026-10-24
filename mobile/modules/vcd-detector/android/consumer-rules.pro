# ONNX Runtime's JNI code constructs these Java classes by name (e.g. NodeInfo(String, ValueInfo)).
# R8 in release builds removed them, aborting the app in OrtSession.getInputInfo. Keep the API intact.
-keep class ai.onnxruntime.** { *; }
-keepclasseswithmembernames class * { native <methods>; }

# WebRTC call bridge: JavaScript calls these by name (vendored LiveCalling kit).
-keepclassmembers class com.hypenex.vcdcalling.WebRtcCallView$Bridge {
    @android.webkit.JavascriptInterface <methods>;
}
