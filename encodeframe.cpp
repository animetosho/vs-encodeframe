#include <VapourSynth4.h>
#include <VSHelper4.h>
#include <cstring>
#include <string>

#include "fpnge/fpnge.h"
#include <turbojpeg.h>

// requires SSE4.1 minimum
// TODO: add AVX2 support
#include <smmintrin.h>
#define MWORD_SIZE 16  // sizeof(__m128i)


/// planar -> interleaved conversion

static inline void copy1x16b(uint8_t* VS_RESTRICT dst, const uint8_t* VS_RESTRICT src0, int width, int bits, bool endianSwap) {
	uint16_t* d16 = reinterpret_cast<uint16_t*>(dst);
	const uint16_t* s0_16 = reinterpret_cast<const uint16_t*>(src0);
	int shl = endianSwap ? (24-bits) : (16-bits);
	int shr = endianSwap ? (bits-8) : (bits*2 - 16);
	__m128i vshl = _mm_set_epi32(0, shr, 0, shl);
	__m128i vshr = _mm_unpackhi_epi64(vshl, vshl);
	
	int x = 0;
	for(; x<width-MWORD_SIZE/2+1; x+=MWORD_SIZE/2) {
		__m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s0_16 + x));
		s0 = _mm_or_si128(_mm_sll_epi16(s0, vshl), _mm_srl_epi16(s0, vshr));
		_mm_store_si128(reinterpret_cast<__m128i*>(d16 + x), s0);
	}
	for(; x<width; x++) {
		d16[x] = (s0_16[x] << shl) | (s0_16[x] >> shr);
	}
}

static inline void interleave2x8b(uint8_t* VS_RESTRICT dst, const uint8_t* VS_RESTRICT src0, const uint8_t* VS_RESTRICT src1, int width) {
	int x = 0;
	for(; x<width-MWORD_SIZE+1; x+=MWORD_SIZE) {
		__m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src0 + x));
		__m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1 + x));
		
		__m128i* d = reinterpret_cast<__m128i*>(dst + x*2);
		_mm_store_si128(d+0, _mm_unpacklo_epi8(s0, s1));
		_mm_store_si128(d+1, _mm_unpackhi_epi8(s0, s1));
	}
	for(; x<width; x++) {
		dst[x*2 +0] = src0[x];
		dst[x*2 +1] = src1[x];
	}
}
static inline void interleave2x16b(uint8_t* VS_RESTRICT dst, const uint8_t* VS_RESTRICT src0, const uint8_t* VS_RESTRICT src1, int width, int bits, bool endianSwap) {
	uint16_t* d16 = reinterpret_cast<uint16_t*>(dst);
	const uint16_t* s0_16 = reinterpret_cast<const uint16_t*>(src0);
	const uint16_t* s1_16 = reinterpret_cast<const uint16_t*>(src1);
	int shl = endianSwap ? (24-bits) : (16-bits);
	int shr = endianSwap ? (bits-8) : (bits*2 - 16);
	__m128i vshl = _mm_set_epi32(0, shr, 0, shl);
	__m128i vshr = _mm_unpackhi_epi64(vshl, vshl);
	
	int x = 0;
	for(; x<width-MWORD_SIZE/2+1; x+=MWORD_SIZE/2) {
		__m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s0_16 + x));
		__m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s1_16 + x));
		
		s0 = _mm_or_si128(_mm_sll_epi16(s0, vshl), _mm_srl_epi16(s0, vshr));
		s1 = _mm_or_si128(_mm_sll_epi16(s1, vshl), _mm_srl_epi16(s1, vshr));
		
		__m128i* d = reinterpret_cast<__m128i*>(d16 + x*2);
		_mm_store_si128(d+0, _mm_unpacklo_epi16(s0, s1));
		_mm_store_si128(d+1, _mm_unpackhi_epi16(s0, s1));
	}
	for(; x<width; x++) {
		d16[x*2 +0] = (s0_16[x] << shl) | (s0_16[x] >> shr);
		d16[x*2 +1] = (s1_16[x] << shl) | (s1_16[x] >> shr);
	}
}
static inline void interleave3x8b(uint8_t* VS_RESTRICT dst, const uint8_t* VS_RESTRICT src0, const uint8_t* VS_RESTRICT src1, const uint8_t* VS_RESTRICT src2, int width) {
	int x = 0;
	__m128i blend1 = _mm_set_epi32(0x0000ff00, 0x00ff0000, 0xff0000ff, 0x0000ff00);
	__m128i blend2 = _mm_bslli_si128(blend1, 1);
	__m128i shuf0 = _mm_set_epi32(0x050a0f04, 0x090e0308, 0x0d02070c, 0x01060b00);
	__m128i shuf1 = _mm_alignr_epi8(shuf0, shuf0, 15);
	__m128i shuf2 = _mm_alignr_epi8(shuf0, shuf0, 14);
	for(; x<width-MWORD_SIZE+1; x+=MWORD_SIZE) {
		__m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src0 + x));
		__m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1 + x));
		__m128i s2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2 + x));
		
		// re-arrange into groups of 3
		s0 = _mm_shuffle_epi8(s0, shuf0);
		s1 = _mm_shuffle_epi8(s1, shuf1);
		s2 = _mm_shuffle_epi8(s2, shuf2);
		
		// blend together
		__m128i d0 = _mm_blendv_epi8(s0, s1, blend1);
		__m128i d1 = _mm_blendv_epi8(s1, s2, blend1);
		__m128i d2 = _mm_blendv_epi8(s2, s0, blend1);
		d0 = _mm_blendv_epi8(d0, s2, blend2);
		d1 = _mm_blendv_epi8(d1, s0, blend2);
		d2 = _mm_blendv_epi8(d2, s1, blend2);
		
		__m128i* d = reinterpret_cast<__m128i*>(dst + x*3);
		_mm_store_si128(d+0, d0);
		_mm_store_si128(d+1, d1);
		_mm_store_si128(d+2, d2);
	}
	for(; x<width; x++) {
		dst[x*3 +0] = src0[x];
		dst[x*3 +1] = src1[x];
		dst[x*3 +2] = src2[x];
	}
}
static inline void interleave3x16b(uint8_t* VS_RESTRICT dst, const uint8_t* VS_RESTRICT src0, const uint8_t* VS_RESTRICT src1, const uint8_t* VS_RESTRICT src2, int width, int bits, bool endianSwap) {
	uint16_t* d16 = reinterpret_cast<uint16_t*>(dst);
	const uint16_t* s0_16 = reinterpret_cast<const uint16_t*>(src0);
	const uint16_t* s1_16 = reinterpret_cast<const uint16_t*>(src1);
	const uint16_t* s2_16 = reinterpret_cast<const uint16_t*>(src2);
	int shl = endianSwap ? (24-bits) : (16-bits);
	int shr = endianSwap ? (bits-8) : (bits*2 - 16);
	__m128i vshl = _mm_set_epi32(0, shr, 0, shl);
	__m128i vshr = _mm_unpackhi_epi64(vshl, vshl);
	
	__m128i shuf0 = _mm_set_epi32(0x0b0a0504, 0x0f0e0908, 0x03020d0c, 0x07060100);
	__m128i shuf1 = _mm_alignr_epi8(shuf0, shuf0, 14);
	__m128i shuf2 = _mm_alignr_epi8(shuf0, shuf0, 12);
	int x = 0;
	for(; x<width-MWORD_SIZE/2+1; x+=MWORD_SIZE/2) {
		__m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s0_16 + x));
		__m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s1_16 + x));
		__m128i s2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s2_16 + x));
		
		s0 = _mm_or_si128(_mm_sll_epi16(s0, vshl), _mm_srl_epi16(s0, vshr));
		s1 = _mm_or_si128(_mm_sll_epi16(s1, vshl), _mm_srl_epi16(s1, vshr));
		s2 = _mm_or_si128(_mm_sll_epi16(s2, vshl), _mm_srl_epi16(s2, vshr));
		
		// re-arrange into groups of 3
		s0 = _mm_shuffle_epi8(s0, shuf0);
		s1 = _mm_shuffle_epi8(s1, shuf1);
		s2 = _mm_shuffle_epi8(s2, shuf2);
		
		// blend together
		__m128i d0 = _mm_blend_epi16(s0, s1, 0b10010010);
		__m128i d1 = _mm_blend_epi16(s2, s0, 0b10010010);
		__m128i d2 = _mm_blend_epi16(s1, s2, 0b10010010);
		d0 = _mm_blend_epi16(d0, s2, 0b00100100);
		d1 = _mm_blend_epi16(d1, s1, 0b00100100);
		d2 = _mm_blend_epi16(d2, s0, 0b00100100);
		
		__m128i* d = reinterpret_cast<__m128i*>(d16 + x*3);
		_mm_store_si128(d+0, d0);
		_mm_store_si128(d+1, d1);
		_mm_store_si128(d+2, d2);
	}
	for(; x<width; x++) {
		d16[x*3 +0] = (s0_16[x] << shl) | (s0_16[x] >> shr);
		d16[x*3 +1] = (s1_16[x] << shl) | (s1_16[x] >> shr);
		d16[x*3 +2] = (s2_16[x] << shl) | (s2_16[x] >> shr);
	}
}
static inline void interleave4x8b(uint8_t* VS_RESTRICT dst, const uint8_t* VS_RESTRICT src0, const uint8_t* VS_RESTRICT src1, const uint8_t* VS_RESTRICT src2, const uint8_t* VS_RESTRICT src3, int width) {
	int x = 0;
	for(; x<width-MWORD_SIZE+1; x+=MWORD_SIZE) {
		__m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src0 + x));
		__m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src1 + x));
		__m128i s2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src2 + x));
		__m128i s3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src3 + x));
		
		__m128i mix0 = _mm_unpacklo_epi8(s0, s1);
		__m128i mix1 = _mm_unpackhi_epi8(s0, s1);
		__m128i mix2 = _mm_unpacklo_epi8(s2, s3);
		__m128i mix3 = _mm_unpackhi_epi8(s2, s3);
		
		__m128i* d = reinterpret_cast<__m128i*>(dst + x*4);
		_mm_store_si128(d+0, _mm_unpacklo_epi16(mix0, mix2));
		_mm_store_si128(d+1, _mm_unpackhi_epi16(mix0, mix2));
		_mm_store_si128(d+2, _mm_unpacklo_epi16(mix1, mix3));
		_mm_store_si128(d+3, _mm_unpackhi_epi16(mix1, mix3));
	}
	for(; x<width; x++) {
		dst[x*4 +0] = src0[x];
		dst[x*4 +1] = src1[x];
		dst[x*4 +2] = src2[x];
		dst[x*4 +3] = src3[x];
	}
}
static inline void interleave4x16b(uint8_t* VS_RESTRICT dst, const uint8_t* VS_RESTRICT src0, const uint8_t* VS_RESTRICT src1, const uint8_t* VS_RESTRICT src2, const uint8_t* VS_RESTRICT src3, int width, int bits, bool endianSwap) {
	uint16_t* d16 = reinterpret_cast<uint16_t*>(dst);
	const uint16_t* s0_16 = reinterpret_cast<const uint16_t*>(src0);
	const uint16_t* s1_16 = reinterpret_cast<const uint16_t*>(src1);
	const uint16_t* s2_16 = reinterpret_cast<const uint16_t*>(src2);
	const uint16_t* s3_16 = reinterpret_cast<const uint16_t*>(src3);
	int shl = endianSwap ? (24-bits) : (16-bits);
	int shr = endianSwap ? (bits-8) : (bits*2 - 16);
	__m128i vshl = _mm_set_epi32(0, shr, 0, shl);
	__m128i vshr = _mm_unpackhi_epi64(vshl, vshl);
	
	int x = 0;
	for(; x<width-MWORD_SIZE/2+1; x+=MWORD_SIZE/2) {
		__m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s0_16 + x));
		__m128i s1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s1_16 + x));
		__m128i s2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s2_16 + x));
		__m128i s3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s3_16 + x));
		
		s0 = _mm_or_si128(_mm_sll_epi16(s0, vshl), _mm_srl_epi16(s0, vshr));
		s1 = _mm_or_si128(_mm_sll_epi16(s1, vshl), _mm_srl_epi16(s1, vshr));
		s2 = _mm_or_si128(_mm_sll_epi16(s2, vshl), _mm_srl_epi16(s2, vshr));
		s3 = _mm_or_si128(_mm_sll_epi16(s3, vshl), _mm_srl_epi16(s3, vshr));
		
		__m128i mix0 = _mm_unpacklo_epi16(s0, s1);
		__m128i mix1 = _mm_unpackhi_epi16(s0, s1);
		__m128i mix2 = _mm_unpacklo_epi16(s2, s3);
		__m128i mix3 = _mm_unpackhi_epi16(s2, s3);
		
		__m128i* d = reinterpret_cast<__m128i*>(d16 + x*4);
		_mm_store_si128(d+0, _mm_unpacklo_epi32(mix0, mix2));
		_mm_store_si128(d+1, _mm_unpackhi_epi32(mix0, mix2));
		_mm_store_si128(d+2, _mm_unpacklo_epi32(mix1, mix3));
		_mm_store_si128(d+3, _mm_unpackhi_epi32(mix1, mix3));
	}
	for(; x<width; x++) {
		d16[x*4 +0] = (s0_16[x] << shl) | (s0_16[x] >> shr);
		d16[x*4 +1] = (s1_16[x] << shl) | (s1_16[x] >> shr);
		d16[x*4 +2] = (s2_16[x] << shl) | (s2_16[x] >> shr);
		d16[x*4 +3] = (s3_16[x] << shl) | (s3_16[x] >> shr);
	}
}


/// VapourSynth function

static void VS_CC encodeFrame(const VSMap* in, VSMap* out, void*, VSCore*, const VSAPI* vsapi) {
	int err = 0;
	
	int no_param = 0;
	int param = vsapi->mapGetIntSaturated(in, "param", 0, &no_param);
	
	std::string imgFormat = vsapi->mapGetData(in, "imgformat", 0, nullptr);
	if(imgFormat != "PNG" && imgFormat != "JPEG") {
		vsapi->mapSetError(out, "EncodeFrame: Format must be PNG/JPEG");
		return;
	}
	
	if(imgFormat == "JPEG") {
		if(no_param) param = 75;
		if(param < 0 || param > 100) {
			vsapi->mapSetError(out, "EncodeFrame: JPEG quality must be between 0 and 100");
			return;
		}
	}
	if(imgFormat == "PNG") {
		if(no_param) param = FPNGE_COMPRESS_LEVEL_DEFAULT;
		if(param < 1 || param > FPNGE_COMPRESS_LEVEL_BEST) {
			#define _STRINGIFY(i) #i
			vsapi->mapSetError(out, "EncodeFrame: PNG level must be between 1 and " _STRINGIFY(FPNGE_COMPRESS_LEVEL_BEST));
			#undef _STRINGIFY
			return;
		}
	}
	
	const VSFrame* frame = vsapi->mapGetFrame(in, "frame", 0, nullptr);
	const VSVideoFormat* fi = vsapi->getVideoFrameFormat(frame);
	
	if((fi->colorFamily != cfRGB && fi->colorFamily != cfGray)
	    || fi->sampleType == stFloat || fi->bytesPerSample > 2 || fi->bitsPerSample < 8)
	{
		vsapi->freeFrame(frame);
		vsapi->mapSetError(out, "EncodeFrame: Only constant format 8-16 bit integer RGB and Grayscale input supported");
		return;
	}
	
	// TODO: TurboJPEG 3 supports >8b precision for JPEGs
	// also consider YUV as a colour source?
	if(imgFormat == "JPEG" && fi->bytesPerSample > 1) {
		vsapi->freeFrame(frame);
		vsapi->mapSetError(out, "EncodeFrame: JPEG only supports 1 byte per sample");
		return;
	}
	
	int width = vsapi->getFrameWidth(frame, 0);
	int height = vsapi->getFrameHeight(frame, 0);
	
	const VSFrame *alpha = vsapi->mapGetFrame(in, "alpha", 0, &err);
	if(alpha) {
		const VSVideoFormat *alphaFi = vsapi->getVideoFrameFormat(alpha);
		
		if(width != vsapi->getFrameWidth(alpha, 0) ||
		   height != vsapi->getFrameHeight(alpha, 0) ||
		   alphaFi->colorFamily != cfGray ||
		   alphaFi->sampleType != fi->sampleType ||
		   alphaFi->bitsPerSample != fi->bitsPerSample ||
		   alphaFi->bytesPerSample != fi->bytesPerSample)
		{
			vsapi->freeFrame(frame);
			vsapi->freeFrame(alpha);
			vsapi->mapSetError(out, "EncodeFrame: Alpha frame dimensions and color depth don't match the main frame");
			return;
		}
		if(imgFormat == "JPEG") {
			vsapi->freeFrame(frame);
			vsapi->freeFrame(alpha);
			vsapi->mapSetError(out, "EncodeFrame: JPEG doesn't support alpha");
			return;
		}
	}
	
	
	/// Interleave colour planes
	bool isGray = fi->colorFamily == cfGray;
	int numChannels = isGray ? 1 : 3;
	if(alpha) numChannels++;
	
	unsigned stride = width * fi->bytesPerSample * numChannels;
	stride = (stride + MWORD_SIZE-1) / MWORD_SIZE * MWORD_SIZE;
	size_t size = stride * height;
	uint8_t* data;
	VSH_ALIGNED_MALLOC(&data, size, MWORD_SIZE);
	
	if(!data) {
		vsapi->freeFrame(frame);
		vsapi->freeFrame(alpha);
		vsapi->mapSetError(out, "EncodeFrame: Failed to allocate intermediary buffer");
		return;
	}
	
	const uint8_t* VS_RESTRICT r = vsapi->getReadPtr(frame, 0);
	const uint8_t* VS_RESTRICT g = nullptr;
	const uint8_t* VS_RESTRICT b = nullptr;
	const uint8_t* VS_RESTRICT a = nullptr;
	ptrdiff_t strideR = vsapi->getStride(frame, 0);
	ptrdiff_t strideG = 0;
	ptrdiff_t strideB = 0;
	ptrdiff_t strideA = 0;
	
	if(alpha) {
		strideA = vsapi->getStride(alpha, 0);
		a = vsapi->getReadPtr(alpha, 0);
	}
	if(numChannels >= 3) {
		strideG = vsapi->getStride(frame, 1);
		g = vsapi->getReadPtr(frame, 1);
		strideB = vsapi->getStride(frame, 2);
		b = vsapi->getReadPtr(frame, 2);
	}
	
	// NOTE: only PNG supports 16b samples, and that must be in big-endian
	if(numChannels == 1) {
		if(fi->bytesPerSample == 1) {
			// straight copy
			vsh::bitblt(data, stride, r, strideR, width, height);
		} else {
			// upsample / endian swap
			for(int y=0; y<height; y++)
				copy1x16b(data + y*stride, r + y*strideR, width, fi->bitsPerSample, true);
		}
	} else if(numChannels == 2) {
		if(fi->bytesPerSample == 1) {
			for(int y=0; y<height; y++)
				interleave2x8b(data + y*stride, r + y*strideR, a + y*strideA, width);
		} else {
			for(int y=0; y<height; y++)
				interleave2x16b(data + y*stride, r + y*strideR, a + y*strideA, width, fi->bitsPerSample, true);
		}
	} else if(numChannels == 3) {
		if(fi->bytesPerSample == 1) {
			for(int y=0; y<height; y++)
				interleave3x8b(data + y*stride, r + y*strideR, g + y*strideG, b + y*strideB, width);
		} else {
			for(int y=0; y<height; y++)
				interleave3x16b(data + y*stride, r + y*strideR, g + y*strideG, b + y*strideB, width, fi->bitsPerSample, true);
		}
	} else { // numChannels == 4
		if(fi->bytesPerSample == 1) {
			for(int y=0; y<height; y++)
				interleave4x8b(data + y*stride, r + y*strideR, g + y*strideG, b + y*strideB, a + y*strideA, width);
		} else {
			for(int y=0; y<height; y++)
				interleave4x16b(data + y*stride, r + y*strideR, g + y*strideG, b + y*strideB, a + y*strideA, width, fi->bitsPerSample, true);
		}
	}
	
	vsapi->freeFrame(frame);
	if(alpha) vsapi->freeFrame(alpha);
	
	
	/// encode PNG/JPEG
	uint8_t* encData;
	size_t encSize;
	
	if(imgFormat == "JPEG") {
		// TODO: support subsampling option
		int subsamp = isGray ? TJSAMP_GRAY : TJSAMP_420;
		encSize = tjBufSize(width, height, subsamp);
		VSH_ALIGNED_MALLOC(&encData, encSize, MWORD_SIZE);
		if(!encData) {
			VSH_ALIGNED_FREE(data);
			vsapi->mapSetError(out, "EncodeFrame: Failed to allocate output buffer");
			return;
		}
		
		tjhandle handle = tjInitCompress();
		if(!handle) {
			VSH_ALIGNED_FREE(data);
			vsapi->mapSetError(out, "EncodeFrame: Failed to allocate libjpeg handle");
			return;
		}
		if(tjCompress2(handle, data, width, stride, height, isGray ? TJPF_GRAY : TJPF_RGB, &encData, &encSize, subsamp, param, TJFLAG_FASTDCT)) {
			vsapi->mapSetError(out, (std::string("EncodeFrame: libjpeg compress error: ") + tjGetErrorStr()).c_str());
			tjDestroy(handle);
			VSH_ALIGNED_FREE(data);
			return;
		}
		tjDestroy(handle);
	} else { // imgFormat == "PNG"
		encSize = FPNGEOutputAllocSize(fi->bytesPerSample, numChannels, width, height);
		VSH_ALIGNED_MALLOC(&encData, encSize, MWORD_SIZE);
		if(!encData) {
			VSH_ALIGNED_FREE(data);
			vsapi->mapSetError(out, "EncodeFrame: Failed to allocate output buffer");
			return;
		}
		struct FPNGEOptions options;
		FPNGEFillOptions(&options, param, 0);
		encSize = FPNGEEncode(fi->bytesPerSample, numChannels, data, width, stride, height, encData, &options);
	}
	VSH_ALIGNED_FREE(data);
	
	/// return encoded data
	vsapi->mapSetData(out, "bytes", reinterpret_cast<char*>(encData), encSize, dtBinary, maReplace);
	VSH_ALIGNED_FREE(encData);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
	vspapi->configPlugin("animetosho.encodeframe", "encodeframe", "VapourSynth EncodeFrame module", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
	vspapi->registerFunction("EncodeFrame", "frame:vframe;imgformat:data;param:int:opt;alpha:vframe:opt;", "bytes:data;", encodeFrame, nullptr, plugin);
}
