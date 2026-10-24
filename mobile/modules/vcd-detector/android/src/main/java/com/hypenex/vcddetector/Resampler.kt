package com.hypenex.vcddetector

import kotlin.math.PI
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Band-limited resampler approximating libsoxr "HQ" (what librosa's res_type="soxr_hq" uses):
 * linear-phase Kaiser-windowed sinc, passband to 91.3% of the lower Nyquist, stopband from 100%,
 * ~120 dB attenuation, output aligned with the input (no delay). Not bit-identical to soxr;
 * differences are measured against the Colab reference export.
 */
object Resampler {
  private const val PASS_END = 0.913
  private const val STOP_BEGIN = 1.0
  private const val ATTEN_DB = 120.0
  private val BETA = 0.1102 * (ATTEN_DB - 8.7)
  private const val MAX_PHASES = 4096

  private fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

  private fun besselI0(x: Double): Double {
    var sum = 1.0; var term = 1.0; var k = 1
    val q = x * x / 4
    while (term > 1e-12 * sum) { term *= q / (k.toDouble() * k); sum += term; k++ }
    return sum
  }

  fun resample(x: FloatArray, inRate: Int, outRate: Int): FloatArray {
    if (inRate == outRate) return x.copyOf()
    val g = gcd(inRate, outRate)
    val up = outRate / g   // L
    val down = inRate / g  // M
    val base = minOf(inRate, outRate).toDouble()
    val fc = 0.5 * (PASS_END + STOP_BEGIN) * base / 2          // cutoff, Hz
    val tw = (STOP_BEGIN - PASS_END) * base / 2                 // transition width, Hz
    val halfSec = (ATTEN_DB - 7.95) / (2.285 * 2 * PI * tw) / 2 // kernel half-length, s
    val k = ceil(halfSec * inRate).toInt()                      // half-length in input samples
    val i0Beta = besselI0(BETA)

    // h in input-sample units: kernel(u) = (2 fc / inRate) * sinc(2 fc u / inRate) * kaiser(u / (halfSec*inRate))
    val halfIn = halfSec * inRate
    val scale = 2 * fc / inRate
    fun kernel(u: Double): Double {
      val r = u / halfIn
      if (r <= -1.0 || r >= 1.0) return 0.0
      val a = scale * u
      val sinc = if (a == 0.0) 1.0 else sin(PI * a) / (PI * a)
      return scale * sinc * besselI0(BETA * sqrt(1 - r * r)) / i0Beta
    }

    val nOut = ceil(x.size.toDouble() * outRate / inRate).toInt()
    val out = FloatArray(nOut)
    val taps = 2 * k + 2
    if (up <= MAX_PHASES) {
      // Polyphase table: tab[p][j] = kernel(K - j + p/L)
      val tab = Array(up) { p -> DoubleArray(taps) { j -> kernel(k - j + p.toDouble() / up) } }
      for (n in 0 until nOut) {
        val pos = n.toLong() * down
        val i0 = (pos / up).toInt()
        val p = (pos % up).toInt()
        val h = tab[p]
        var acc = 0.0
        val start = i0 - k
        val jFrom = maxOf(0, -start)
        val jTo = minOf(taps, x.size - start)
        for (j in jFrom until jTo) acc += x[start + j] * h[j]
        out[n] = acc.toFloat()
      }
    } else {
      for (n in 0 until nOut) {
        val t = n.toDouble() * inRate / outRate
        val i0 = floor(t).toInt()
        var acc = 0.0
        for (i in maxOf(0, i0 - k)..minOf(x.size - 1, i0 + k + 1)) acc += x[i] * kernel(t - i)
        out[n] = acc.toFloat()
      }
    }
    return out
  }
}
