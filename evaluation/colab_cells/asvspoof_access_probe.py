# ASVspoof access probe (diagnostic only): a few tiny Range requests, NO archive download.
# Run in a fresh Colab cell; paste the whole output back.
import subprocess, urllib.request, urllib.error

URLS = {
    "landing": "https://datashare.ed.ac.uk/handle/10283/3336",
    "bitstream_api": "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269",
    "LA.zip_content": "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content",
    "LA.zip_handle": "https://datashare.ed.ac.uk/bitstream/handle/10283/3336/LA.zip",
}
UAS = {"python-urllib": None, "curl": "curl/8.5.0",
       "browser": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36"}
for name, url in URLS.items():
    for ua_name, ua in UAS.items():
        h = {"Range": "bytes=0-3"} if "LA.zip" in name else {}
        if ua:
            h["User-Agent"] = ua
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=30) as r:
                r.read(4)
                print(f"{name:15s} {ua_name:13s} HTTP {r.status} content-range={r.headers.get('content-range')}")
        except urllib.error.HTTPError as e:
            keep = {k: v for k, v in e.headers.items() if k.lower() in ("server", "cf-ray", "x-cache", "via", "retry-after")}
            print(f"{name:15s} {ua_name:13s} HTTP {e.code} {e.reason} {keep} body={e.read(200)!r}")
        except Exception as e:
            print(f"{name:15s} {ua_name:13s} {type(e).__name__}: {e}")
print(subprocess.run(["curl", "-sS", "-L", "-r", "0-3", "-o", "/dev/null", "-w", "curl final HTTP %{http_code} via %{url_effective}\n",
                      URLS["LA.zip_content"]], capture_output=True, text=True).stdout)
print("egress IP:", subprocess.run(["curl", "-sS", "https://api.ipify.org"], capture_output=True, text=True).stdout)
