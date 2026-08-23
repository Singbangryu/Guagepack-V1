`ifndef GAUGEPACK_VFU_EXTERNAL_OPS_VH
`define GAUGEPACK_VFU_EXTERNAL_OPS_VH

// =============================================================================
// GaugePack VFU TOP operation contract
// =============================================================================
// "Internal ops are not exposed" is an RTL interface rule, not documentation
// secrecy:
//
//   TOP -> VFU controller : VFU_TOP_OP_W-bit function-level op
//   VFU controller -> CORE16/reduction blocks : private 4-bit VFU_OP_* micro-op
//
// TOP must never drive or observe the private 4-bit micro-op field.  The VFU
// controller expands each TOP op into the internal sequence documented below.
// Internal macro definitions and encodings remain in vfu_defs.vh.
//
// -----------------------------------------------------------------------------
// VFU_RQ
// -----------------------------------------------------------------------------
// Internal CORE16 sequence:
//   `VFU_OP_RQ
//
// -----------------------------------------------------------------------------
// VFU_GELU
// -----------------------------------------------------------------------------
// Internal CORE16 sequence:
//   `VFU_OP_GELU
//
// -----------------------------------------------------------------------------
// VFU_SM
// -----------------------------------------------------------------------------
// Internal controller sequence:
//   1. ROWMAX reduction phase          (vfu_softmax state; no CORE16 micro-op)
//   2. `VFU_OP_QEXP
//   3. ROWSUM reduction phase          (vfu_softmax state; no CORE16 micro-op)
//   4. `VFU_OP_SM_RECIP_RAW
//
// This operation ends after E, L, and reciprocal R are ready.  It does not
// perform N/L because numerator N is produced later by the PMPU PV pass.
//
// -----------------------------------------------------------------------------
// VFU_SM_NL
// -----------------------------------------------------------------------------
// Issued only after PMPU PV has produced numerator N.
//
// Internal CORE16 sequence:
//   `VFU_OP_SM_CONTEXT
//
// Arithmetic:
//   context = N/L = RNE((N * R) >> 23)
//
// -----------------------------------------------------------------------------
// VFU_LN
// -----------------------------------------------------------------------------
// Internal controller sequence:
//   1. `VFU_OP_RQ_RES
//   2. `VFU_OP_LN_MOMENT_INIT
//   3. `VFU_OP_LN_MOMENT_ACC          (repeat for remaining beats)
//   4. `VFU_OP_LN_D
//   5. `VFU_OP_LN_RSQRT
//   6. `VFU_OP_LN_NORM
//   7. `VFU_OP_LN_AFFINE
//
// Encodings 3'b101 through 3'b111 are reserved.
// =============================================================================

`define VFU_TOP_OP_W            3

`define VFU_RQ                  3'b000
`define VFU_GELU                3'b001
`define VFU_SM                  3'b010
`define VFU_SM_NL               3'b011
`define VFU_LN                  3'b100

`endif
