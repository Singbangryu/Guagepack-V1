`ifndef GAUGEPACK_VFU_EXTERNAL_OPS_VH
`define GAUGEPACK_VFU_EXTERNAL_OPS_VH

// =============================================================================
// GaugePack VFU external operation contract
// =============================================================================
// Only these coarse operations are visible at the VFU boundary.  Detailed
// micro-op encodings remain private to the VFU controller.  The comments below
// describe the functional phases only; they are not externally issued opcodes.
//
// VFU_NON_LINEAR_RQ
//   - Generic requantization
//   - GELU Direct32/Q-Map
//   - Page/metadata selects the pointwise transform.
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
// =============================================================================

`define VFU_EXTERNAL_OP_W       2

`define VFU_NON_LINEAR_RQ       2'b00
`define VFU_SM                  2'b01
`define VFU_SM_NL               2'b10
`define VFU_LN                  2'b11

`endif
