#include <VapourSynth4.h>
#include <VSHelper4.h>
#include <cstring>
#include <string>

#include "fpnge/fpnge.h"
#ifdef HAVE_JPEG
#include <turbojpeg.h>
#endif
#ifdef HAVE_WEBP
#include <webp/encode.h>
#endif

// requires SSE4.1 minimum
#ifdef __AVX2__
# include <immintrin.h>
# define MWORD_SIZE 32  // sizeof(__m256i)
# define MM(f) _mm256_##f
# define MMSI(f) _mm256_##f##_si256
# define MIVEC __m256i
# define BCAST128 _mm256_broadcastsi128_si256
# define SWAP_MID64(x) _mm256_permute4x64_epi64(x, _MM_SHUFFLE(3,1,2,0))
#else
# include <smmintrin.h>
# define MWORD_SIZE 16  // sizeof(__m128i)
# define MM(f) _mm_##f
# define MMSI(f) _mm_##f##_si128
# define MIVEC __m128i
# define BCAST128(v) (v)
# define SWAP_MID64(x) (x)
#endif

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
		MIVEC s0 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(s0_16 + x));
		s0 = MMSI(or)(MM(sll_epi16)(s0, vshl), MM(srl_epi16)(s0, vshr));
		MMSI(store)(reinterpret_cast<MIVEC*>(d16 + x), s0);
	}
	for(; x<width; x++) {
		d16[x] = (s0_16[x] << shl) | (s0_16[x] >> shr);
	}
}

static inline void interleave2x8b(uint8_t* VS_RESTRICT dst, const uint8_t* VS_RESTRICT src0, const uint8_t* VS_RESTRICT src1, int width) {
	int x = 0;
	for(; x<width-MWORD_SIZE+1; x+=MWORD_SIZE) {
		MIVEC s0 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(src0 + x));
		MIVEC s1 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(src1 + x));
		
		s0 = SWAP_MID64(s0);
		s1 = SWAP_MID64(s1);
		
		MIVEC* d = reinterpret_cast<MIVEC*>(dst + x*2);
		MMSI(store)(d+0, MM(unpacklo_epi8)(s0, s1));
		MMSI(store)(d+1, MM(unpackhi_epi8)(s0, s1));
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
		MIVEC s0 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(s0_16 + x));
		MIVEC s1 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(s1_16 + x));
		
		s0 = MMSI(or)(MM(sll_epi16)(s0, vshl), MM(srl_epi16)(s0, vshr));
		s1 = MMSI(or)(MM(sll_epi16)(s1, vshl), MM(srl_epi16)(s1, vshr));
		
		s0 = SWAP_MID64(s0);
		s1 = SWAP_MID64(s1);
		
		MIVEC* d = reinterpret_cast<MIVEC*>(d16 + x*2);
		MMSI(store)(d+0, MM(unpacklo_epi16)(s0, s1));
		MMSI(store)(d+1, MM(unpackhi_epi16)(s0, s1));
	}
	for(; x<width; x++) {
		d16[x*2 +0] = (s0_16[x] << shl) | (s0_16[x] >> shr);
		d16[x*2 +1] = (s1_16[x] << shl) | (s1_16[x] >> shr);
	}
}
static inline void interleave3x8b(uint8_t* VS_RESTRICT dst, const uint8_t* VS_RESTRICT src0, const uint8_t* VS_RESTRICT src1, const uint8_t* VS_RESTRICT src2, int width) {
	int x = 0;
	MIVEC blend1 = BCAST128(_mm_set_epi32(0x0000ff00, 0x00ff0000, 0xff0000ff, 0x0000ff00));
	MIVEC blend2 = MMSI(slli)(blend1, 1);
	MIVEC shuf0 = BCAST128(_mm_set_epi32(0x050a0f04, 0x090e0308, 0x0d02070c, 0x01060b00));
	MIVEC shuf1 = MM(alignr_epi8)(shuf0, shuf0, 15);
	MIVEC shuf2 = MM(alignr_epi8)(shuf0, shuf0, 14);
	for(; x<width-MWORD_SIZE+1; x+=MWORD_SIZE) {
		MIVEC s0 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(src0 + x));
		MIVEC s1 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(src1 + x));
		MIVEC s2 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(src2 + x));
		
		// re-arrange into groups of 3
		s0 = MM(shuffle_epi8)(s0, shuf0);
		s1 = MM(shuffle_epi8)(s1, shuf1);
		s2 = MM(shuffle_epi8)(s2, shuf2);
		
		// blend together
		MIVEC d0 = MM(blendv_epi8)(s0, s1, blend1);
		MIVEC d1 = MM(blendv_epi8)(s1, s2, blend1);
		MIVEC d2 = MM(blendv_epi8)(s2, s0, blend1);
		d0 = MM(blendv_epi8)(d0, s2, blend2);
		d1 = MM(blendv_epi8)(d1, s0, blend2);
		d2 = MM(blendv_epi8)(d2, s1, blend2);
		
#ifdef __AVX2__
		s0 = _mm256_permute2x128_si256(d0, d1, 0x20);
		s1 = _mm256_permute2x128_si256(d2, d0, 0x30);
		s2 = _mm256_permute2x128_si256(d1, d2, 0x31);
		d0 = s0;
		d1 = s1;
		d2 = s2;
#endif
		
		MIVEC* d = reinterpret_cast<MIVEC*>(dst + x*3);
		MMSI(store)(d+0, d0);
		MMSI(store)(d+1, d1);
		MMSI(store)(d+2, d2);
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
	
	MIVEC shuf0 = BCAST128(_mm_set_epi32(0x0b0a0504, 0x0f0e0908, 0x03020d0c, 0x07060100));
	MIVEC shuf1 = MM(alignr_epi8)(shuf0, shuf0, 14);
	MIVEC shuf2 = MM(alignr_epi8)(shuf0, shuf0, 12);
	int x = 0;
	for(; x<width-MWORD_SIZE/2+1; x+=MWORD_SIZE/2) {
		MIVEC s0 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(s0_16 + x));
		MIVEC s1 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(s1_16 + x));
		MIVEC s2 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(s2_16 + x));
		
		s0 = MMSI(or)(MM(sll_epi16)(s0, vshl), MM(srl_epi16)(s0, vshr));
		s1 = MMSI(or)(MM(sll_epi16)(s1, vshl), MM(srl_epi16)(s1, vshr));
		s2 = MMSI(or)(MM(sll_epi16)(s2, vshl), MM(srl_epi16)(s2, vshr));
		
		// re-arrange into groups of 3
		s0 = MM(shuffle_epi8)(s0, shuf0);
		s1 = MM(shuffle_epi8)(s1, shuf1);
		s2 = MM(shuffle_epi8)(s2, shuf2);
		
		// blend together
		MIVEC d0 = MM(blend_epi16)(s0, s1, 0b10010010);
		MIVEC d1 = MM(blend_epi16)(s2, s0, 0b10010010);
		MIVEC d2 = MM(blend_epi16)(s1, s2, 0b10010010);
		d0 = MM(blend_epi16)(d0, s2, 0b00100100);
		d1 = MM(blend_epi16)(d1, s1, 0b00100100);
		d2 = MM(blend_epi16)(d2, s0, 0b00100100);
		
#ifdef __AVX2__
		s0 = _mm256_permute2x128_si256(d0, d1, 0x20);
		s1 = _mm256_permute2x128_si256(d2, d0, 0x30);
		s2 = _mm256_permute2x128_si256(d1, d2, 0x31);
		d0 = s0;
		d1 = s1;
		d2 = s2;
#endif
		
		MIVEC* d = reinterpret_cast<MIVEC*>(d16 + x*3);
		MMSI(store)(d+0, d0);
		MMSI(store)(d+1, d1);
		MMSI(store)(d+2, d2);
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
		MIVEC s0 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(src0 + x));
		MIVEC s1 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(src1 + x));
		MIVEC s2 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(src2 + x));
		MIVEC s3 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(src3 + x));
		
		MIVEC mix0 = MM(unpacklo_epi8)(s0, s1);
		MIVEC mix1 = MM(unpackhi_epi8)(s0, s1);
		MIVEC mix2 = MM(unpacklo_epi8)(s2, s3);
		MIVEC mix3 = MM(unpackhi_epi8)(s2, s3);
		
		s0 = MM(unpacklo_epi16)(mix0, mix2);
		s1 = MM(unpackhi_epi16)(mix0, mix2);
		s2 = MM(unpacklo_epi16)(mix1, mix3);
		s3 = MM(unpackhi_epi16)(mix1, mix3);
		
#ifdef __AVX2__
		mix0 = _mm256_permute2x128_si256(s0, s1, 0x20);
		mix1 = _mm256_permute2x128_si256(s2, s3, 0x20);
		mix2 = _mm256_permute2x128_si256(s0, s1, 0x31);
		mix3 = _mm256_permute2x128_si256(s2, s3, 0x31);
		s0 = mix0;
		s1 = mix1;
		s2 = mix2;
		s3 = mix3;
#endif
		
		MIVEC* d = reinterpret_cast<MIVEC*>(dst + x*4);
		MMSI(store)(d+0, s0);
		MMSI(store)(d+1, s1);
		MMSI(store)(d+2, s2);
		MMSI(store)(d+3, s3);
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
		MIVEC s0 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(s0_16 + x));
		MIVEC s1 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(s1_16 + x));
		MIVEC s2 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(s2_16 + x));
		MIVEC s3 = MMSI(loadu)(reinterpret_cast<const MIVEC*>(s3_16 + x));
		
		s0 = MMSI(or)(MM(sll_epi16)(s0, vshl), MM(srl_epi16)(s0, vshr));
		s1 = MMSI(or)(MM(sll_epi16)(s1, vshl), MM(srl_epi16)(s1, vshr));
		s2 = MMSI(or)(MM(sll_epi16)(s2, vshl), MM(srl_epi16)(s2, vshr));
		s3 = MMSI(or)(MM(sll_epi16)(s3, vshl), MM(srl_epi16)(s3, vshr));
		
		MIVEC mix0 = MM(unpacklo_epi16)(s0, s1);
		MIVEC mix1 = MM(unpackhi_epi16)(s0, s1);
		MIVEC mix2 = MM(unpacklo_epi16)(s2, s3);
		MIVEC mix3 = MM(unpackhi_epi16)(s2, s3);
		
		s0 = MM(unpacklo_epi32)(mix0, mix2);
		s1 = MM(unpackhi_epi32)(mix0, mix2);
		s2 = MM(unpacklo_epi32)(mix1, mix3);
		s3 = MM(unpackhi_epi32)(mix1, mix3);
		
#ifdef __AVX2__
		mix0 = _mm256_permute2x128_si256(s0, s1, 0x20);
		mix1 = _mm256_permute2x128_si256(s2, s3, 0x20);
		mix2 = _mm256_permute2x128_si256(s0, s1, 0x31);
		mix3 = _mm256_permute2x128_si256(s2, s3, 0x31);
		s0 = mix0;
		s1 = mix1;
		s2 = mix2;
		s3 = mix3;
#endif
		
		MIVEC* d = reinterpret_cast<MIVEC*>(d16 + x*4);
		MMSI(store)(d+0, s0);
		MMSI(store)(d+1, s1);
		MMSI(store)(d+2, s2);
		MMSI(store)(d+3, s3);
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
	
	int no_quality = 0, no_effort = 0;
	int quality = vsapi->mapGetInt(in, "quality", 0, &no_quality);
	int effort = vsapi->mapGetInt(in, "effort", 0, &no_effort);
	if(no_quality) quality = 75;
	
	std::string imgFormat = vsapi->mapGetData(in, "imgformat", 0, nullptr);
	if(imgFormat != "PNG"
#ifdef HAVE_JPEG
	 && imgFormat != "JPEG"
#endif
#ifdef HAVE_WEBP
	 && imgFormat != "WEBP-VP8" && imgFormat != "WEBP"
#endif
	) {
		vsapi->mapSetError(out, "EncodeFrame: Format must be PNG"
#ifdef HAVE_JPEG
	 "/JPEG"
#endif
#ifdef HAVE_WEBP
	 "/WEBP/WEBP-VP8"
#endif
		);
		return;
	}
	
	if(quality < 0 || quality > 100) {
		vsapi->mapSetError(out, "EncodeFrame: quality must be between 0 and 100");
		return;
	}
	if(imgFormat == "PNG") {
		if(no_effort) effort = FPNGE_COMPRESS_LEVEL_DEFAULT;
		if(effort < 1 || effort > FPNGE_COMPRESS_LEVEL_BEST) {
			#define _STR_HELPER(i) #i
			#define _STRINGIFY(i) _STR_HELPER(i)
			vsapi->mapSetError(out, "EncodeFrame: PNG effort must be between 1 and " _STRINGIFY(FPNGE_COMPRESS_LEVEL_BEST));
			#undef _STR_HELPER
			#undef _STRINGIFY
			return;
		}
	}
	if(imgFormat == "WEBP" || imgFormat == "WEBP-VP8") {
		if(no_effort) effort = 4;
		if(effort < 1 || effort > 6) {
			vsapi->mapSetError(out, "EncodeFrame: WebP effort must be between 1 and 6");
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
	if((imgFormat == "JPEG" || imgFormat == "WEBP" || imgFormat == "WEBP-VP8") && fi->bytesPerSample > 1) {
		vsapi->freeFrame(frame);
		vsapi->mapSetError(out, "EncodeFrame: JPEG/WebP only supports 1 byte per sample");
		return;
	}
	if((imgFormat == "WEBP" || imgFormat == "WEBP-VP8") && fi->colorFamily == cfGray) {
		vsapi->freeFrame(frame);
		vsapi->mapSetError(out, "EncodeFrame: WebP doesn't support grayscale - please convert to RGB(A) instead");
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
	
	
	/// encode to image format
	uint8_t* encData;
	size_t encSize;
	
	if(imgFormat == "JPEG") {
#ifdef HAVE_JPEG
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
		if(tjCompress2(handle, data, width, stride, height, isGray ? TJPF_GRAY : TJPF_RGB, &encData, &encSize, subsamp, quality, TJFLAG_FASTDCT)) {
			vsapi->mapSetError(out, (std::string("EncodeFrame: libjpeg compress error: ") + tjGetErrorStr()).c_str());
			tjDestroy(handle);
			VSH_ALIGNED_FREE(data);
			return;
		}
		tjDestroy(handle);
#endif
	} else if(imgFormat == "WEBP" || imgFormat == "WEBP-VP8") {
#ifdef HAVE_WEBP
		WebPConfig config;
		WebPPicture pic;
		if(!WebPConfigPreset(&config, WEBP_PRESET_DEFAULT, quality) ||
		   !WebPPictureInit(&pic)) {
			VSH_ALIGNED_FREE(data);
			vsapi->mapSetError(out, "EncodeFrame: Failed to initialize WebP");
			return;
		}
		config.lossless = (imgFormat == "WEBP" ? 1 : 0);
		config.method = effort;
		// TODO: support other options?
		
		if(!WebPValidateConfig(&config)) {
			VSH_ALIGNED_FREE(data);
			vsapi->mapSetError(out, "EncodeFrame: Invalid WebP configuration");
			return;
		}
		
		// TODO: support YUV input?
		pic.width = width;
		pic.height = height;
		if(!WebPPictureAlloc(&pic)) {
			VSH_ALIGNED_FREE(data);
			vsapi->mapSetError(out, "EncodeFrame: Failed to allocate WebP output");
			return;
		}
		// TODO: consider writing directly to WebPPicture to avoid this importing business
		if(numChannels == 3)
			WebPPictureImportRGB(&pic, data, stride);
		if(numChannels == 4)
			WebPPictureImportRGBA(&pic, data, stride);
		VSH_ALIGNED_FREE(data);
		
		WebPMemoryWriter wrt;
		WebPMemoryWriterInit(&wrt);
		pic.writer = WebPMemoryWrite;
		pic.custom_ptr = &wrt;
		
		int ok = WebPEncode(&config, &pic);
		WebPPictureFree(&pic);
		if(!ok) {
			std::string error("EncodeFrame: Failed to encode WebP: ");
			switch(pic.error_code) {
				case VP8_ENC_ERROR_OUT_OF_MEMORY:
					error += "memory error allocating objects"; break;
				case VP8_ENC_ERROR_BITSTREAM_OUT_OF_MEMORY:
					error += "memory error while flushing bits"; break;
				case VP8_ENC_ERROR_NULL_PARAMETER:
					error += "a pointer parameter is NULL"; break;
				case VP8_ENC_ERROR_INVALID_CONFIGURATION:
					error += "configuration is invalid"; break;
				case VP8_ENC_ERROR_BAD_DIMENSION:
					error += "picture has invalid width/height"; break;
				case VP8_ENC_ERROR_PARTITION0_OVERFLOW:
					error += "partition is bigger than 512k"; break;
				case VP8_ENC_ERROR_PARTITION_OVERFLOW:
					error += "partition is bigger than 16M"; break;
				case VP8_ENC_ERROR_BAD_WRITE:
					error += "error while flushing bytes"; break;
				case VP8_ENC_ERROR_FILE_TOO_BIG:
					error += "file is bigger than 4G"; break;
				case VP8_ENC_ERROR_USER_ABORT:
					error += "abort request by user"; break;
				default:
					error += "unknown code (" + std::to_string(pic.error_code) + ")";
			}
			WebPMemoryWriterClear(&wrt);
			vsapi->mapSetError(out, error.c_str());
			return;
		}
		
		// we get the pointer, instead of allocating it ourself, so return from here and skip the PNG/JPEG path
		vsapi->mapSetData(out, "bytes", reinterpret_cast<char*>(wrt.mem), wrt.size, dtBinary, maReplace);
		WebPMemoryWriterClear(&wrt);
		return;
#endif
	} else { // imgFormat == "PNG"
		encSize = FPNGEOutputAllocSize(fi->bytesPerSample, numChannels, width, height);
		VSH_ALIGNED_MALLOC(&encData, encSize, MWORD_SIZE);
		if(!encData) {
			VSH_ALIGNED_FREE(data);
			vsapi->mapSetError(out, "EncodeFrame: Failed to allocate output buffer");
			return;
		}
		struct FPNGEOptions options;
		FPNGEFillOptions(&options, effort, 0);
		encSize = FPNGEEncode(fi->bytesPerSample, numChannels, data, width, stride, height, encData, &options);
	}
	VSH_ALIGNED_FREE(data);
	
	/// return encoded data
	vsapi->mapSetData(out, "bytes", reinterpret_cast<char*>(encData), encSize, dtBinary, maReplace);
	VSH_ALIGNED_FREE(encData);
}


VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin *plugin, const VSPLUGINAPI *vspapi) {
	vspapi->configPlugin("animetosho.encodeframe", "encodeframe", "VapourSynth EncodeFrame module", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
	vspapi->registerFunction("EncodeFrame", "frame:vframe;imgformat:data;quality:int:opt;effort:int:opt;alpha:vframe:opt;", "bytes:data;", encodeFrame, nullptr, plugin);
}
