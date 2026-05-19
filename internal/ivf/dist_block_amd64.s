//go:build amd64

#include "textflag.h"

// func distBlockDiff(q *[14]int16, blocks unsafe.Pointer, blockOff uint32, dim int, diffs *[8]int32)
//
// Computes q[dim] - blocks[blockOff + dim*16 + v*2] for v=0..7.
// Stores 8 int32 results in diffs[0..7].
//
// ABI0 frame layout (stack-based, 5 × 8-byte args = 40):
//   +0(FP):  q        *[14]int16 (pointer, 8 bytes)
//   +8(FP):  blocks   unsafe.Pointer (pointer, 8 bytes)
//  +16(FP):  blockOff uint32 (4 bytes, padded to 8)
//  +24(FP):  dim      int (8 bytes)
//  +32(FP):  diffs    *[8]int32 (pointer, 8 bytes)
//
// Uses VMOVDQU (unaligned) because IVF3 block offsets are NOT guaranteed
// 16-byte aligned: HeaderSz + K*(CentroidSz+BBoxSz+ClusterEntrySz)
// = 20 + K*92. For K=4096: 20+376832=376852, 376852%16=4.
TEXT ·distBlockDiff(SB), NOSPLIT, $0-40
	// Load parameters from ABI0 stack frame
	MOVQ q+0(FP), DI         // DI = &q[0] (pointer to 14-int16 array)
	MOVQ blocks+8(FP), SI    // SI = &blocks[0] (base pointer)
	MOVL blockOff+16(FP), R8 // R8 = blockOff (uint32, zero-extended to 64 bits)
	MOVQ dim+24(FP), CX      // CX = dim (int, 0..13)
	MOVQ diffs+32(FP), DX    // DX = &diffs[0] (output pointer)

	// Step 1: Load q[dim] (int16 at DI + dim*2), zero-extend to int32
	SHLQ $1, CX              // CX = dim * 2 (byte offset in q)
	MOVWQSX (DI)(CX*1), AX   // AX = int32(q[dim]), sign-extended from int16

	// Step 2: Broadcast q[dim] to all 8 lanes of YMM Y0 (int32)
	MOVQ AX, X0              // Place q[dim] in lower 32 bits of X0
	VPBROADCASTD X0, Y0       // Y0 = [qd, qd, qd, qd, qd, qd, qd, qd]

	// Step 3: Compute &blocks[blockOff + dim*16]
	// CX = dim*2 from step 1, multiply by 8 to get dim*16
	SHLQ $3, CX              // CX = dim * 16
	ADDQ R8, CX              // CX = blockOff + dim*16
	ADDQ SI, CX              // CX = &blocks[blockOff + dim*16]

	// Step 4: Load 8×int16 → XMM X1 (128 bits, unaligned OK on Haswell)
	VMOVDQU (CX), X1

	// Step 5: Widen int16 → int32 (128→256 bits, sign-extended)
	VPMOVSXWD X1, Y1

	// Step 6: Subtract broadcast q[dim] from all 8 vectors
	VPSUBD Y1, Y0, Y2         // Y2 = q[dim] - block_dim_values

	// Step 7: Store 8×int32 results
	VMOVDQU Y2, (DX)

	VZEROUPPER
	RET

// func distBlockFused(q *[14]int16, blocks unsafe.Pointer, blockOff uint32,
//                     sums *[8]int64, maxDist int64) (aliveMask uint32)
//
// Computes full 14-dimension squared Euclidean distances for 8 vectors
// in a single fused kernel. Float32 dual-accumulator with int64 storage.
//
// ABI0 frame layout (6 args + 1 return = 48 bytes):
//   q        +0(FP)  *[14]int16      (8 bytes)
//   blocks   +8(FP)  unsafe.Pointer   (8 bytes)
//   blockOff +16(FP) uint32           (4 bytes, padded to 8)
//   sums     +24(FP) *[8]int64        (8 bytes)
//   maxDist  +32(FP) int64            (8 bytes)
//   ret      +40(FP) uint32           (4 bytes, padded to 48)
//
// Register allocation:
//   AX = &q[0]
//   BX = &blocks[blockOff]
//   DI = &sums[0]
//   R10 = maxDist
//   Y0 = even-dim float32 accumulator (dims 0,2,4,6,8,10,12)
//   Y4 = odd-dim float32 accumulator  (dims 1,3,5,7,9,11,13)
//   Y1 = scratch: block dim values int32→float32
//   Y2 = scratch: q[dim] broadcast int16→int32→float32 → diff
//   Y3 = checkpoint merged accumulator / comparison mask
TEXT ·distBlockFused(SB), NOSPLIT, $0-44
	MOVQ q+0(FP), AX           // AX = &q[0]
	MOVQ blocks+8(FP), BX      // BX = blocks base pointer
	MOVL blockOff+16(FP), R9   // R9 = blockOff
	ADDQ R9, BX                // BX = &blocks[blockOff]
	MOVQ sums+24(FP), DI       // DI = &sums[0]
	MOVQ maxDist+32(FP), R10   // R10 = maxDist

	// HW prefetch next block (at offset +240 from current)
	PREFETCHT0 256(BX)
	PREFETCHT0 320(BX)
	PREFETCHT0 384(BX)

	// Init float32 accumulators
	VXORPS Y0, Y0, Y0          // even-dim acc = 0
	VXORPS Y4, Y4, Y4          // odd-dim acc = 0

	// === Dims 0..7, dual-accumulator ===
	//
	// dim 0 → Y0 (even)
	VPMOVSXWD    (BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW (AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	// dim 1 → Y4 (odd)
	VPMOVSXWD    16(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW 2(AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y4

	// dim 2 → Y0 (even)
	VPMOVSXWD    32(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW 4(AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	// dim 3 → Y4 (odd)
	VPMOVSXWD    48(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW 6(AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y4

	// dim 4 → Y0 (even)
	VPMOVSXWD    64(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW 8(AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	// dim 5 → Y4 (odd)
	VPMOVSXWD    80(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW 10(AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y4

	// dim 6 → Y0 (even)
	VPMOVSXWD    96(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW 12(AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	// dim 7 → Y4 (odd)
	VPMOVSXWD    112(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW 14(AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y4

	// === Checkpoint at dim 8: merge accumulators, compare vs maxDist ===
	VADDPS    Y4, Y0, Y3       // Y3 = even + odd (partial, 8 dims)
	VCVTSI2SSQ R10, X2, X2      // X2[0] = float32(maxDist)
	VBROADCASTSS X2, Y2         // Y2 = [float32(maxDist) × 8]
	VCMPPS    $0x01, Y2, Y3, Y3 // Y3 = (Y3 < Y2)? (mask for alive lanes)
	VMOVMSKPS Y3, R8            // R8 = bitmask of lanes where partial < maxDist
	TESTL     R8, R8
	JZ        earlyDead

	// === Dims 8..13, continue dual-accumulator ===
	//
	// dim 8 → Y0 (even)
	VPMOVSXWD    128(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW 16(AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	// dim 9 → Y4 (odd)
	VPMOVSXWD    144(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW 18(AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y4

	// dim 10 → Y0 (even)
	VPMOVSXWD    160(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW 20(AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	// dim 11 → Y4 (odd)
	VPMOVSXWD    176(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW 22(AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y4

	// dim 12 → Y0 (even)
	VPMOVSXWD    192(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW 24(AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y0

	// dim 13 → Y4 (odd)
	VPMOVSXWD    208(BX), Y1
	VCVTDQ2PS    Y1, Y1
	VPBROADCASTW 26(AX), X2
	VPMOVSXWD    X2, Y2
	VCVTDQ2PS    Y2, Y2
	VSUBPS       Y1, Y2, Y2
	VFMADD231PS  Y2, Y2, Y4

	// === Final merge, convert float32→int32→int64, store to sums ===
	JMP finalMerge

earlyDead:
	// All lanes dead at dim 8 checkpoint — store partial sums (unused)
	// and return aliveMask = 0. The fused merge is done so
	// the same conversion+store code runs for both paths.

finalMerge:
	VADDPS    Y4, Y0, Y0       // merge even+odd → Y0 (float32)

	// Convert float32 → int32 → int64
	VCVTPS2DQ Y0, Y0           // float32×8 → int32×8

	// Lanes 0-3: sums[0..3]
	VPMOVSXDQ X0, Y1           // int32×4 → int64×4
	VMOVDQU   Y1, (DI)         // store sums[0..3]

	// Lanes 4-7: sums[4..7]
	VEXTRACTI128 $1, Y0, X2     // extract upper 4 int32
	VPMOVSXDQ   X2, Y1          // int32×4 → int64×4
	VMOVDQU     Y1, 32(DI)      // store sums[4..7] (offset 32 = 4×8)

	// Build aliveMask: for each lane v, set bit v if sums[v] < maxDist
	XORL R8, R8
	CMPQ R10, 0(DI)
	JLE next0
	ORL $0x01, R8
next0:
	CMPQ R10, 8(DI)
	JLE next1
	ORL $0x02, R8
next1:
	CMPQ R10, 16(DI)
	JLE next2
	ORL $0x04, R8
next2:
	CMPQ R10, 24(DI)
	JLE next3
	ORL $0x08, R8
next3:
	CMPQ R10, 32(DI)
	JLE next4
	ORL $0x10, R8
next4:
	CMPQ R10, 40(DI)
	JLE next5
	ORL $0x20, R8
next5:
	CMPQ R10, 48(DI)
	JLE next6
	ORL $0x40, R8
next6:
	CMPQ R10, 56(DI)
	JLE next7
	ORL $0x80, R8
next7:

	MOVL R8, aliveMask+40(FP)
	VZEROUPPER
	RET
