# ONNX Runtime's JNI code constructs these Java classes by name (e.g. NodeInfo(String, ValueInfo)).
# R8 in release builds removed them, aborting the app in OrtSession.getInputInfo. Keep the API intact.
-keep class ai.onnxruntime.** { *; }
-keepclasseswithmembernames class * { native <methods>; }
