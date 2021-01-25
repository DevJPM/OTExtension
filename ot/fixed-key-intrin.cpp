#include "fixed-key-intrin.h"
#include "ot-ext.h"

#include <immintrin.h>

__attribute__((target("aes,sse4.1"))) __attribute__((always_inline))
static void expandAESKey(__m128i userkey, uint8_t* storagePointer)
{
	// this uses the fast AES key expansion (i.e. not using keygenassist) from
	// https://www.intel.com/content/dam/doc/white-paper/advanced-encryption-standard-new-instructions-set-paper.pdf
	// page 37

	__m128i temp1, temp2, temp3, globAux;
	const __m128i shuffle_mask =
		_mm_set_epi32(0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d, 0x0c0f0e0d);
	const __m128i con3 = _mm_set_epi32(0x07060504, 0x07060504, 0x0ffffffff, 0x0ffffffff);
	__m128i rcon;
	temp1 = userkey;
	rcon = _mm_set_epi32(1, 1, 1, 1);
	_mm_storeu_si128((__m128i*)(storagePointer + 0 * 16), temp1);
	for (int i = 1; i <= 8; i++) {
		temp2 = _mm_shuffle_epi8(temp1, shuffle_mask);
		temp2 = _mm_aesenclast_si128(temp2, rcon);
		rcon = _mm_slli_epi32(rcon, 1);
		globAux = _mm_slli_epi64(temp1, 32);
		temp1 = _mm_xor_si128(globAux, temp1);
		globAux = _mm_shuffle_epi8(temp1, con3);
		temp1 = _mm_xor_si128(globAux, temp1);
		temp1 = _mm_xor_si128(temp2, temp1);
		_mm_storeu_si128((__m128i*)(storagePointer + i * 16), temp1);
	}
	rcon = _mm_set_epi32(0x1b, 0x1b, 0x1b, 0x1b);
	temp2 = _mm_shuffle_epi8(temp1, shuffle_mask);
	temp2 = _mm_aesenclast_si128(temp2, rcon);
	rcon = _mm_slli_epi32(rcon, 1);
	globAux = _mm_slli_epi64(temp1, 32);
	temp1 = _mm_xor_si128(globAux, temp1);
	globAux = _mm_shuffle_epi8(temp1, con3);
	temp1 = _mm_xor_si128(globAux, temp1);
	temp1 = _mm_xor_si128(temp2, temp1);
	_mm_storeu_si128((__m128i*)(storagePointer + 9 * 16), temp1);
	temp2 = _mm_shuffle_epi8(temp1, shuffle_mask);
	temp2 = _mm_aesenclast_si128(temp2, rcon);
	globAux = _mm_slli_epi64(temp1, 32);
	temp1 = _mm_xor_si128(globAux, temp1);
	globAux = _mm_shuffle_epi8(temp1, con3);
	temp1 = _mm_xor_si128(globAux, temp1);
	temp1 = _mm_xor_si128(temp2, temp1);
	_mm_storeu_si128((__m128i*)(storagePointer + 10 * 16), temp1);
}

template<size_t width> __attribute__((target("vaes,avx512f"))) __attribute__((always_inline))
static void EncryptVAES(__m512i data[width], const __m512i round_keys[11]) {

	for (size_t w = 0; w < width; ++w)
	{
		data[w] = _mm512_xor_si512(data[w], round_keys[0]);
	}

	for (size_t r = 1; r < 10; ++r) {
		for (size_t w = 0; w < width; ++w) {
			data[w] = _mm512_aesenc_epi128(data[w], round_keys[r]);
		}
	}

	for (size_t w = 0; w < width; ++w)
	{
		data[w] = _mm512_aesenclast_epi128(data[w], round_keys[10]);
	}
}

template<size_t width> __attribute__((target("aes,sse4.1"))) __attribute__((always_inline))
static void EncryptAESNI(__m128i data[width], const __m128i round_keys[11]) {
	for (size_t w = 0; w < width; ++w) {
		data[w] = _mm_xor_si128(data[w], round_keys[0]);
	}

	for (size_t r = 1; r < 10; ++r) {
		for (size_t w = 0; w < width; ++w) {
			data[w] = _mm_aesenc_si128(data[w], round_keys[r]);
		}
	}

	for (size_t w = 0; w < width; ++w) {
		data[w] = _mm_aesenclast_si128(data[w], round_keys[10]);
	}
}

template<size_t width> __attribute__((target("vaes,avx512f"))) __attribute__((always_inline))
static void CTREncryptVAES(const uint64_t base_counter, BYTE*& outbuf, uint64_t& currentNumBlocks, BYTE* expanded_keys) {
	const uint64_t remaining = currentNumBlocks % (4*width);
	const uint64_t processing = currentNumBlocks - remaining;

	__m512i counter = _mm512_set_epi64(0, base_counter + 3, 0, base_counter + 2, 0, base_counter + 1, 0, base_counter + 0);
	const __m512i counter_diff = _mm512_set_epi64(0, 4, 0, 4, 0, 4, 0, 4);

	__m512i round_keys[11];
	for (size_t k = 0; k < 11; ++k) {
		__m128i buffer = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&expanded_keys[16 * k]));
		round_keys[k] = _mm512_broadcast_i32x4(buffer);
	}

	for (uint64_t b = 0; b < processing; b += 4*width) {
		__m512i data[width];

		for (size_t w = 0; w < width; ++w) {
			data[w] = counter;
			counter = _mm512_add_epi64(counter, counter_diff);
		}

		EncryptVAES<width>(data, round_keys);

		for (size_t w = 0; w < width; ++w) {
			_mm512_storeu_si512(reinterpret_cast<__m512i*>(&outbuf[64 * w]), data[w]);
		}

		outbuf += 64 * width;
		currentNumBlocks -= 4*width;
	}
}

template<size_t width> __attribute__((target("aes,sse4.1"))) __attribute__((always_inline))
static void CTREncryptAESNI(const uint64_t base_counter, BYTE*& outbuf, uint64_t& currentNumBlocks, BYTE* expanded_keys) {
	const uint64_t remaining = currentNumBlocks % width;
	const uint64_t processing = currentNumBlocks - remaining;

	__m128i counter = _mm_set_epi64x(0,base_counter);
	const __m128i counter_diff = _mm_set_epi64x(0,1);

	__m128i round_keys[11];
	for (size_t k = 0; k < 11; ++k)
		round_keys[k] = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&expanded_keys[16*k]));

	for (uint64_t b = 0; b < processing; b += width) {
		__m128i data[width];

		for (size_t w = 0; w < width; ++w) {
			data[w] = counter;
			counter = _mm_add_epi64(counter, counter_diff);
		}

		EncryptAESNI<width>(data, round_keys);

		for (size_t w = 0; w < width; ++w) {
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&outbuf[16 * w]), data[w]);
		}

		outbuf += 16 * width;
		currentNumBlocks -= width;
	}
}

template<size_t width> __attribute__((target("vaes,avx512f"))) __attribute__((always_inline))
static void FixedKeyHashingIntrinSndVAES(BYTE**& outbufs,
	BYTE*& Q,
	const BYTE* U,
	uint64_t& base_id,
	uint32_t& base_u,
	const uint64_t numOTs,
	const uint32_t numSendVals,
	const __m128i round_keys[11])
{
	const uint64_t processed = base_id * numSendVals + base_u;
	const uint64_t leftForProcessing = numOTs * numSendVals - processed;
	const uint64_t remaining = leftForProcessing % (4*width);
	const uint64_t processing = leftForProcessing - remaining;

	const __m128i uval = _mm_loadu_si128(reinterpret_cast<const __m128i*>(U));

	__m512i vaes_round_keys[11];
	for (int i = 0; i < 11; ++i)
		vaes_round_keys[i] = _mm512_broadcast_i32x4(round_keys[i]);

	for (uint64_t b = 0; b < processing; b += 4*width) {
		__m512i data[width];
		__m512i whitening[width];
		uint8_t* targetPointer[4*width];

		for (size_t w = 0; w < width; ++w)
		{
			__m128i whitening_buffer[4];
			__m128i data_buffer[4];
			
			for (size_t k = 0; k < 4; ++k) {
				__m128i qval = _mm_loadu_si128(reinterpret_cast<__m128i*>(Q));

				if (base_u == 1) {
					qval = _mm_xor_si128(qval, uval);
					_mm_storeu_si128(reinterpret_cast<__m128i*>(Q), qval);
				}
				data_buffer[k] = _mm_set_epi64x(0, base_id);

				whitening_buffer[k] = qval;
				data_buffer[k] = _mm_xor_si128(qval, data_buffer[k]);

				targetPointer[4*w+k] = outbufs[base_u];
				outbufs[base_u] += 16;

				base_u++;
				if (base_u >= numSendVals) {
					base_u = 0;
					base_id++;
					Q += 16;
				}
			}
			
			data[w] = _mm512_inserti32x4(data[w], data_buffer[0], 0);
			data[w] = _mm512_inserti32x4(data[w], data_buffer[1], 1);
			data[w] = _mm512_inserti32x4(data[w], data_buffer[2], 2);
			data[w] = _mm512_inserti32x4(data[w], data_buffer[3], 3);
			whitening[w] = _mm512_inserti32x4(whitening[w], whitening_buffer[0], 0);
			whitening[w] = _mm512_inserti32x4(whitening[w], whitening_buffer[1], 1);
			whitening[w] = _mm512_inserti32x4(whitening[w], whitening_buffer[2], 2);
			whitening[w] = _mm512_inserti32x4(whitening[w], whitening_buffer[3], 3);
		}

		EncryptVAES<width>(data, vaes_round_keys);

		for (size_t w = 0; w < width; ++w) {
			data[w] = _mm512_xor_si512(data[w], whitening[w]);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(targetPointer[4 * w + 0]), _mm512_extracti32x4_epi32(data[w], 0));
			_mm_storeu_si128(reinterpret_cast<__m128i*>(targetPointer[4 * w + 1]), _mm512_extracti32x4_epi32(data[w], 1));
			_mm_storeu_si128(reinterpret_cast<__m128i*>(targetPointer[4 * w + 2]), _mm512_extracti32x4_epi32(data[w], 2));
			_mm_storeu_si128(reinterpret_cast<__m128i*>(targetPointer[4 * w + 3]), _mm512_extracti32x4_epi32(data[w], 3));
		}
	}
}

template<size_t width> __attribute__((target("aes,sse4.1"))) __attribute__((always_inline))
static void FixedKeyHashingIntrinSndAESNI(BYTE**& outbufs,
	BYTE*& Q,
	const BYTE* U,
	uint64_t& base_id,
	uint32_t& base_u,
	const uint64_t numOTs,
	const uint32_t numSendVals,
	const __m128i round_keys[11])
{
	const uint64_t processed = base_id * numSendVals + base_u;
	const uint64_t leftForProcessing = numOTs * numSendVals - processed;
	const uint64_t remaining = leftForProcessing % width;
	const uint64_t processing = leftForProcessing - remaining;

	const __m128i uval = _mm_loadu_si128(reinterpret_cast<const __m128i*>(U));

	for (uint64_t b = 0; b < processing; b += width) {
		__m128i data[width];
		__m128i whitening[width];
		uint8_t* targetPointer[width];

		for (size_t w = 0; w < width; ++w)
		{
			__m128i qval = _mm_loadu_si128(reinterpret_cast<__m128i*>(Q));

			if (base_u == 1) {
				qval = _mm_xor_si128(qval, uval);
				_mm_storeu_si128(reinterpret_cast<__m128i*>(Q), qval);
			}
			data[w] = _mm_set_epi64x(0, base_id);

			whitening[w] = qval;
			data[w] = _mm_xor_si128(qval, data[w]);

			targetPointer[w] = outbufs[base_u];
			outbufs[base_u] += 16;

			base_u++;
			if (base_u >= numSendVals) {
				base_u = 0;
				base_id++;
				Q += 16;
			}
		}

		EncryptAESNI<width>(data, round_keys);

		for (size_t w = 0; w < width; ++w) {
			data[w] = _mm_xor_si128(data[w], whitening[w]);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(targetPointer[w]), data[w]);
		}
	}
}

// width in 512-bit words
template<size_t width> __attribute__((target("vaes,avx512f"))) __attribute__((always_inline))
static void FixedKeyHashingIntrinRecVAES(BYTE*& outbuf, BYTE*& inbuf, uint64_t& base_id, uint64_t& num_blocks, const __m128i round_keys[11])
{
	const uint64_t remaining = num_blocks % (4 * width);
	const uint64_t processing = num_blocks - remaining;

	__m512i counter = _mm512_set_epi64(0, base_id + 3, 0, base_id + 2, 0, base_id + 1, 0, base_id + 0);
	const __m512i counter_diff = _mm512_set_epi64(0, 4, 0, 4, 0, 4, 0, 4);

	__m512i vaes_round_keys[11];
	for (int i = 0; i < 11; ++i)
		vaes_round_keys[i] = _mm512_broadcast_i32x4(round_keys[i]);

	for (uint64_t b = 0; b < processing; b += 4 * width) {
		__m512i data[width];
		__m512i whitening[width];

		for (size_t w = 0; w < width; ++w)
		{
			data[w] = _mm512_loadu_si512(reinterpret_cast<__m512i*>(&inbuf[64 * w]));
			whitening[w] = data[w];
			data[w] = _mm512_xor_si512(data[w], counter);
			counter = _mm512_add_epi64(counter, counter_diff);
		}

		EncryptVAES<width>(data, vaes_round_keys);

		for (size_t w = 0; w < width; ++w) {
			data[w] = _mm512_xor_si512(data[w], whitening[w]);
			_mm512_storeu_si512(reinterpret_cast<__m512i*>(&outbuf[64 * w]), data[w]);
		}

		inbuf += width * 64;
		outbuf += width * 64;
		base_id += width * 4;
		num_blocks -= width * 4;
	}
}

template<size_t width> __attribute__((target("aes,sse4.1"))) __attribute__((always_inline))
static void FixedKeyHashingIntrinRecAESNI(BYTE*& outbuf, BYTE*& inbuf, uint64_t& base_id, uint64_t& num_blocks, const __m128i round_keys[11])
{
	const uint64_t remaining = num_blocks % width;
	const uint64_t processing = num_blocks - remaining;

	__m128i counter = _mm_set_epi64x(0, base_id);
	const __m128i counter_diff = _mm_set_epi64x(0, 1);

	for (uint64_t b = 0; b < processing; b += width) {
		__m128i data[width];
		__m128i whitening[width];

		for (size_t w = 0; w < width; ++w)
		{
			data[w] = _mm_loadu_si128(reinterpret_cast<__m128i*>(&inbuf[16 * w]));
			whitening[w] = data[w];
			data[w] = _mm_xor_si128(data[w], counter);
			counter = _mm_add_epi64(counter, counter_diff);
		}

		EncryptAESNI<width>(data, round_keys);

		for (size_t w = 0; w < width; ++w) {
			data[w] = _mm_xor_si128(data[w], whitening[w]);
			_mm_storeu_si128(reinterpret_cast<__m128i*>(&outbuf[16 * w]), data[w]);
		}

		inbuf += width * 16;
		outbuf += width * 16;
		base_id += width;
		num_blocks -= width;
	}
}

__attribute__((target("aes,sse4.1,vaes,avx512f")))
void FixedKeyHashingIntrinRec(BYTE* outbuf, BYTE* inbuf, uint64_t base_id, uint64_t num_blocks)
{
	__m128i round_keys[11];
	__m128i fixed_key = _mm_loadu_si128(reinterpret_cast<const __m128i*>(fixed_key_aes_seed));
	expandAESKey(fixed_key, reinterpret_cast<uint8_t*>(round_keys));

	FixedKeyHashingIntrinRecVAES<4>(outbuf, inbuf, base_id, num_blocks, round_keys);
	FixedKeyHashingIntrinRecAESNI<1>(outbuf, inbuf, base_id, num_blocks, round_keys);

	//FixedKeyHashingIntrinRecAESNI<8>(outbuf, inbuf, base_id, num_blocks, round_keys);
	//FixedKeyHashingIntrinRecAESNI<1>(outbuf, inbuf, base_id, num_blocks, round_keys);
}

__attribute__((target("aes,sse4.1,vaes,avx512f")))
void FixedKeyHashingIntrinSnd(BYTE** outbufs, BYTE* Q, BYTE* U, uint64_t base_id, uint64_t numOTs, uint64_t numSendVals)
{
	__m128i round_keys[11];
	__m128i fixed_key = _mm_loadu_si128(reinterpret_cast<const __m128i*>(fixed_key_aes_seed));
	expandAESKey(fixed_key, reinterpret_cast<uint8_t*>(round_keys));

	uint32_t base_u = 0;
	FixedKeyHashingIntrinSndVAES<4>(outbufs, Q,U, base_id, base_u, numOTs, numSendVals, round_keys);
	FixedKeyHashingIntrinSndAESNI<1>(outbufs, Q,U, base_id, base_u, numOTs, numSendVals, round_keys);
}

__attribute__((target("aes,sse4.1")))
void ParallelKeySchedule(ROUND_KEYS* expanded_keys, BYTE* in_keys, uint32_t num_keys) {
	for (uint32_t i = 0; i < num_keys; ++i, in_keys += 16, expanded_keys++) {
		__m128i userkey = _mm_loadu_si128(reinterpret_cast<const __m128i*>(in_keys));
		expandAESKey(userkey, expanded_keys->keys);
	}
}

__attribute__((target("aes,sse4.1,vaes,avx512f")))
void ParallelEncryption(uint64_t base_counter, BYTE* outbuf, uint64_t num_blocks_per_key, uint32_t num_keys, ROUND_KEYS* expanded_keys)
{
	for (uint32_t i = 0; i < num_keys; ++i, expanded_keys++) {
		uint64_t currentNumBlocks = num_blocks_per_key;
		CTREncryptVAES<4>(base_counter, outbuf, currentNumBlocks, expanded_keys->keys);
		CTREncryptAESNI<1>(base_counter, outbuf, currentNumBlocks, expanded_keys->keys);
	}
}