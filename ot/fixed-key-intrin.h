#ifndef __FIXED_KEY_INTRIN_H
#define __FIXED_KEY_INTRIN_H

#include <ENCRYPTO_utils/typedefs.h>
#include <ENCRYPTO_utils/cbitvector.h>

struct ROUND_KEYS {
	BYTE keys[11 * 16];
};

void FixedKeyHashingIntrinRec(BYTE* outbuf, BYTE* inbuf, uint64_t base_id, uint64_t num_blocks);
void FixedKeyHashingIntrinSnd(BYTE** outbufs, BYTE* Q, BYTE* U, uint64_t base_id, uint64_t numOTs, uint64_t numSendVals);

void ParallelKeySchedule(ROUND_KEYS* expanded_keys, BYTE* in_keys, uint32_t num_keys);
void ParallelEncryption(uint64_t base_counter, BYTE* outbuf, uint64_t num_blocks_per_key, uint32_t num_keys, ROUND_KEYS* expanded_keys);

#endif