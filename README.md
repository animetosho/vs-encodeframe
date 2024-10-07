This is a simple VapourSynth plugin which encodes a `VideoFrame` to a JPEG ([libjpeg-turbo](https://libjpeg-turbo.org/)) or PNG ([fpnge](https://github.com/veluca93/fpnge)). It is built primarily for [Anime Tosho’s frame server](https://github.com/animetosho/frame-server), but can also be useful as a fast image exporter (alternative to imwri).

## Requirements

* VapourSynth R65 or later (earlier versions have a bug which breaks returned bytes data)
* x86 CPU with SSE4.1 support (required by fpnge)
* TurboJPEG

## Building

```
meson setup build
ninja -C build
ninja -C build install
```

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

encodeframe.EncodeFrame(frame: VideoFrame, imgformat: string [, param: int] [, alpha: VideoFrame=None])
------------------------------------------------------------------

Converts a VideoFrame (*frame*) to the format specified by *imgformat* (`"PNG"` or `"JPEG"`) and returns the result as a *bytes* object.

Optionally accepts a grayscale VideoFrame (*alpha*) for PNG. *param* is either a JPEG quality level (0-100, default 75) or fpnge PNG compression level (1-5, default 4).

Note that colourspace conversion is not performed, so *frame* must be in either an RGB or Grayscale colourspace. If *alpha* is supplied, it must have the same colour depth as *frame*.
PNG supports 8 to 16-bit samples, whilst JPEG only allows 8-bit samples. 9 to 15-bit samples will be upsampled to 16-bit.

# See Also

[vsfpng](https://github.com/Mikewando/vsfpng): only supports RGB24/RGB32 PNG output to files, but uses the other fast PNG encoder, fpng

License
=======

This module is Public Domain or [CC0](https://creativecommons.org/publicdomain/zero/1.0/legalcode) (or equivalent) if PD isn’t recognised.

fpnge itself is licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)