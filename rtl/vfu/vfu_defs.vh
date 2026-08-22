`ifndef GAUGEPACK_VFU_DEFS_VH
`define GAUGEPACK_VFU_DEFS_VH

`define VFU_OP_RQ              4'h0
`define VFU_OP_RQ_RES          4'h1
`define VFU_OP_GELU            4'h2
`define VFU_OP_QEXP            4'h3
`define VFU_OP_SM_RECIP_RAW    4'h4
`define VFU_OP_SM_CONTEXT      4'h5
`define VFU_OP_LN_MOMENT_INIT  4'h6
`define VFU_OP_LN_MOMENT_ACC   4'h7
`define VFU_OP_LN_D            4'h8
`define VFU_OP_LN_RSQRT        4'h9
`define VFU_OP_LN_NORM         4'hA
`define VFU_OP_LN_AFFINE       4'hB

`endif
