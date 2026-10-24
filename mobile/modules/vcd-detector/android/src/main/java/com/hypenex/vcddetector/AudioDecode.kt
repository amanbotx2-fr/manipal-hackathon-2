package com.hypenex.vcddetector

import android.content.Context
import android.media.AudioFormat
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class Decoded(
  val sampleRate: Int,
  val channels: Int,
  val frames: Int,
  val mono: FloatArray,
  val decoder: String,
  val notes: Map<String, Any?> = emptyMap(),
)

/**
 * Decodes to float32 at the native rate, then averages channels (like dhwani_infer.to_mono_16k).
 * WAV is parsed directly with soundfile's int-to-float scaling; everything else (m4a/AAC, mp3,
 * ogg/opus, 3gp/amr) goes through MediaExtractor + MediaCodec with 16-bit PCM output, matching
 * audioread's s16 path used by librosa in Colab.
 */
object AudioDecode {
  private const val MAX_BYTES = 64 * 1024 * 1024

  class Unreadable(message: String) : Exception(message)

  fun decode(context: Context, uri: Uri): Decoded {
    val head = ByteArray(12)
    val n = context.contentResolver.openInputStream(uri)?.use { it.read(head) }
      ?: throw Unreadable("cannot open $uri")
    if (n <= 0) throw Unreadable("empty audio (0 bytes)")
    val isWav = n >= 12 && String(head, 0, 4, Charsets.US_ASCII) == "RIFF" &&
      String(head, 8, 4, Charsets.US_ASCII) == "WAVE"
    val d = if (isWav) decodeWav(readAll(context, uri)) else decodeWithMediaCodec(context, uri)
    if (d.frames == 0) throw Unreadable("empty audio (0 samples)")
    if (d.mono.any { !it.isFinite() }) throw Unreadable("audio contains NaN/Inf samples")
    return d
  }

  private fun readAll(context: Context, uri: Uri): ByteArray {
    val out = ByteArrayOutputStream()
    context.contentResolver.openInputStream(uri)!!.use { input ->
      val buf = ByteArray(1 shl 16)
      while (true) {
        val r = input.read(buf); if (r < 0) break
        out.write(buf, 0, r)
        if (out.size() > MAX_BYTES) throw Unreadable("file larger than ${MAX_BYTES / 1048576} MB")
      }
    }
    return out.toByteArray()
  }

  private fun toMono(interleaved: FloatArray, channels: Int): FloatArray {
    if (channels == 1) return interleaved
    val frames = interleaved.size / channels
    return FloatArray(frames) { f ->
      var s = 0.0
      for (c in 0 until channels) s += interleaved[f * channels + c]
      (s / channels).toFloat()
    }
  }

  fun decodeWav(bytes: ByteArray): Decoded {
    val bb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    var pos = 12
    var fmtTag = -1; var channels = 0; var rate = 0; var bits = 0
    var dataOff = -1; var dataLen = 0
    while (pos + 8 <= bytes.size) {
      val id = String(bytes, pos, 4, Charsets.US_ASCII)
      val len = bb.getInt(pos + 4).toLong() and 0xffffffffL
      val body = pos + 8
      if (id == "fmt ") {
        fmtTag = bb.getShort(body).toInt() and 0xffff
        channels = bb.getShort(body + 2).toInt()
        rate = bb.getInt(body + 4)
        bits = bb.getShort(body + 14).toInt()
        if (fmtTag == 0xFFFE && len >= 26) fmtTag = bb.getShort(body + 24).toInt() and 0xffff // WAVE_FORMAT_EXTENSIBLE subformat
      } else if (id == "data") {
        dataOff = body
        dataLen = minOf(len, (bytes.size - body).toLong()).toInt()
        break
      }
      pos = body + len.toInt() + (len.toInt() and 1)
    }
    if (fmtTag < 0 || dataOff < 0 || channels <= 0 || rate <= 0) throw Unreadable("malformed WAV header")
    val bps = bits / 8
    val total = dataLen / bps
    val samples = FloatArray(total - total % channels)
    when {
      fmtTag == 1 && bits == 16 -> for (i in samples.indices) samples[i] = bb.getShort(dataOff + 2 * i) / 32768f
      fmtTag == 1 && bits == 24 -> for (i in samples.indices) {
        val o = dataOff + 3 * i
        val v = (bytes[o].toInt() and 0xff) or ((bytes[o + 1].toInt() and 0xff) shl 8) or (bytes[o + 2].toInt() shl 16)
        samples[i] = v / 8388608f
      }
      fmtTag == 1 && bits == 32 -> for (i in samples.indices) samples[i] = (bb.getInt(dataOff + 4 * i) / 2147483648.0).toFloat()
      fmtTag == 1 && bits == 8 -> for (i in samples.indices) samples[i] = ((bytes[dataOff + i].toInt() and 0xff) - 128) / 128f
      fmtTag == 3 && bits == 32 -> for (i in samples.indices) samples[i] = bb.getFloat(dataOff + 4 * i)
      else -> throw Unreadable("unsupported WAV encoding (format $fmtTag, $bits-bit)")
    }
    return Decoded(rate, channels, samples.size / channels, toMono(samples, channels), "wav-parser",
      mapOf("bits" to bits, "format_tag" to fmtTag))
  }

  fun decodeWithMediaCodec(context: Context, uri: Uri): Decoded {
    val ex = MediaExtractor()
    try {
      ex.setDataSource(context, uri, null)
    } catch (e: Exception) {
      ex.release(); throw Unreadable("unreadable audio (${e.javaClass.simpleName}: ${e.message})")
    }
    var codec: MediaCodec? = null
    try {
      val track = (0 until ex.trackCount).firstOrNull {
        ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
      } ?: throw Unreadable("no audio track")
      ex.selectTrack(track)
      val fmt = ex.getTrackFormat(track)
      val mime = fmt.getString(MediaFormat.KEY_MIME)!!
      val notes = linkedMapOf<String, Any?>("mime" to mime,
        "track_sample_rate" to fmt.getIntegerOr(MediaFormat.KEY_SAMPLE_RATE),
        "track_channels" to fmt.getIntegerOr(MediaFormat.KEY_CHANNEL_COUNT),
        "track_duration_us" to if (fmt.containsKey(MediaFormat.KEY_DURATION)) fmt.getLong(MediaFormat.KEY_DURATION) else null,
        "encoder_delay" to fmt.getIntegerOr("encoder-delay"),
        "encoder_padding" to fmt.getIntegerOr("encoder-padding"))
      codec = MediaCodec.createDecoderByType(mime)
      codec.configure(fmt, null, null, 0)
      codec.start()
      notes["codec"] = codec.name

      var rate = fmt.getIntegerOr(MediaFormat.KEY_SAMPLE_RATE) ?: 0
      var channels = fmt.getIntegerOr(MediaFormat.KEY_CHANNEL_COUNT) ?: 1
      var encoding = AudioFormat.ENCODING_PCM_16BIT
      var pcm = FloatArray(1 shl 18)
      var count = 0
      val info = MediaCodec.BufferInfo()
      var inputDone = false
      var outputDone = false
      var idle = 0
      while (!outputDone) {
        if (!inputDone) {
          val ii = codec.dequeueInputBuffer(10_000)
          if (ii >= 0) {
            val buf = codec.getInputBuffer(ii)!!
            val size = ex.readSampleData(buf, 0)
            if (size < 0) {
              codec.queueInputBuffer(ii, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
              inputDone = true
            } else {
              codec.queueInputBuffer(ii, 0, size, ex.sampleTime, 0)
              ex.advance()
            }
          }
        }
        val oi = codec.dequeueOutputBuffer(info, 10_000)
        when {
          oi == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
            val of = codec.outputFormat
            rate = of.getIntegerOr(MediaFormat.KEY_SAMPLE_RATE) ?: rate
            channels = of.getIntegerOr(MediaFormat.KEY_CHANNEL_COUNT) ?: channels
            encoding = of.getIntegerOr(MediaFormat.KEY_PCM_ENCODING) ?: AudioFormat.ENCODING_PCM_16BIT
          }
          oi >= 0 -> {
            idle = 0
            val out = codec.getOutputBuffer(oi)!!.order(ByteOrder.nativeOrder())
            out.position(info.offset); out.limit(info.offset + info.size)
            val n = if (encoding == AudioFormat.ENCODING_PCM_FLOAT) info.size / 4 else info.size / 2
            if (count + n > pcm.size) pcm = pcm.copyOf(maxOf(pcm.size * 2, count + n))
            if (encoding == AudioFormat.ENCODING_PCM_FLOAT) {
              val fb = out.asFloatBuffer(); fb.get(pcm, count, n)
            } else {
              val sb = out.asShortBuffer()
              for (i in 0 until n) pcm[count + i] = sb.get() / 32768f
            }
            count += n
            codec.releaseOutputBuffer(oi, false)
            if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
          }
          else -> if (inputDone && ++idle > 500) outputDone = true // decoder stalled after EOS
        }
      }
      notes["pcm_encoding"] = if (encoding == AudioFormat.ENCODING_PCM_FLOAT) "float" else "s16"
      val samples = pcm.copyOf(count - count % channels)
      return Decoded(rate, channels, samples.size / channels, toMono(samples, channels), "mediacodec", notes)
    } catch (e: Unreadable) {
      throw e
    } catch (e: Exception) {
      throw Unreadable("unreadable audio (${e.javaClass.simpleName}: ${e.message})")
    } finally {
      try { codec?.stop() } catch (_: Exception) {}
      codec?.release()
      ex.release()
    }
  }

  private fun MediaFormat.getIntegerOr(key: String): Int? =
    if (containsKey(key)) try { getInteger(key) } catch (_: Exception) { null } else null
}
