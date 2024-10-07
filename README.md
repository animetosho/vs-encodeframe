This is a simple VapourSynth plugin which encodes a `VideoFrame` to a JPEG ([libjpeg-turbo](https://libjpeg-turbo.org/)), PNG ([fpnge](https://github.com/veluca93/fpnge)) or WebP ([libwebp](https://github.com/webmproject/libwebp/tree/main)). It is built primarily for [Anime Tosho’s frame server](https://github.com/animetosho/frame-server), but can also be useful as a fast image exporter (alternative to imwri).

## Requirements

* VapourSynth R65 or later (earlier versions have a bug which breaks returned bytes data)
* x86 CPU with SSE4.1 support (required by fpnge)
* TurboJPEG (optional)
* libwebp (optional)

## Building

```
meson setup build
ninja -C build
ninja -C build install
```

If TurboJPEG or libwebp isn't found, respective JPEG/WebP support will be disabled.

Note: fpnge is only built with SSE4.1 support by default. Add `-Disa=avx2` to the first command above to set AVX2 as the baseline.

# Example Usage

Write the first frame of a video to a PNG file:

```python
import vapoursynth as vs

# open input
clip = vs.core.bs.VideoSource("some_video.mkv")
# convert to RGB24
clip = vs.core.resize.Bicubic(clip, format=vs.RGB24)

# encode first frame to PNG
with clip.get_frame(0) as frame:
	data = vs.core.encodeframe.EncodeFrame(frame, "PNG")

# write PNG to file
with open("first_frame.png", "wb") as f:
	f.write(data)
```

API
===

encodeframe.EncodeFrame(frame: VideoFrame, imgformat: string [, quality: int] [, effort: int] [, alpha: VideoFrame=None])
------------------------------------------------------------------

Converts a VideoFrame (*frame*) to the format specified by *imgformat* (`"PNG"`, `"JPEG"`, `"WEBP"` or `"WEBP-VP8"`) and returns the result as a *bytes* object.  
Note that `"WEBP"` is lossless WebP whilst `"WEBP-VP8"` is lossy WebP.

Optionally accepts a grayscale VideoFrame (*alpha*) for PNG/WebP.  
*quality* is a lossy quality level (0-100, default 75) and has a different meaning for lossless WebP. Ignored for PNG.  
*effort* is a WebP or fpnge PNG compression level (1-5 for PNG or 1-6 for WebP, default 4). Ignored for JPEG.

Note that *frame* must be in either an RGB or Grayscale colourspace. If *alpha* is supplied, it must have the same colour depth as *frame*.  
PNG supports 8 to 16-bit samples, whilst JPEG/WebP only allows 8-bit samples. 9 to 15-bit samples will be upsampled to 16-bit.  
WebP doesn't support Grayscale input.

# See Also

[vsfpng](https://github.com/Mikewando/vsfpng): only supports RGB24/RGB32 PNG output to files, but uses the other fast PNG encoder, fpng

License
=======

This module is Public Domain or [CC0](https://creativecommons.org/publicdomain/zero/1.0/legalcode) (or equivalent) if PD isn’t recognised.

fpnge itself is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)