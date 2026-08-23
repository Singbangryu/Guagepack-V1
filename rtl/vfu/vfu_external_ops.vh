`ifndef GAUGEPACK_VFU_EXTERNAL_OPS_VH
`define GAUGEPACK_VFU_EXTERNAL_OPS_VH

// =============================================================================
// GaugePack VFU external operation contract
// =============================================================================
// Only these function-level operations are visible at the VFU boundary.
// Detailed micro-op encodings remain private to the VFU controller.  The
// comments below describe internal functional phases only; they are not
// externally issued opcodes.
//
// VFU_RQ
//   - Generic requantization only
//
// VFU_GELU
//   - GELU Direct32/Q-Map only
//
// VFU_SM
//   - Rowmax ingest/reduction
//   - QEXP
//   - Rowsum ingest/reduction
//   - Reciprocal generation for L
//
// VFU_SM_NL
//   - Terminal Softmax context apply after PMPU PV produces N
//   - Computes context = N/L
//   - Separate from VFU_SM because N is available much later.
//
// VFU_LN
//   - Main requantization + residual add
//   - MomentPack statistics
//   - D = 128Q - S^2
//   - Reciprocal square root
//   - Normalize
//   - Affine + final requantization
//
// Never route the internal micro-op field across the VFU boundary.
// Encodings 3'b101 through 3'b111 are reserved.
// =============================================================================

`define VFU_EXTERNAL_OP_W       3

`define VFU_RQ                  3'b000
`define VFU_GELU                3'b001
`define VFU_SM                  3'b010
`define VFU_SM_NL               3'b011
`define VFU_LN                  3'b100

`endif
