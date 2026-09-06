`timescale 1ns/1ps

// Draft combinational ACT16 residual read request.
// Inputs are already aligned to the S0 beat; this module adds no registers.
// mt_i is padded token count / 16 (legal 1..4), with token_tile_i < mt_i.
// Address units are 128-bit words. The 14-bit result preserves overflow for
// the memory adapter to check before applying its bank-specific address width.
// This module generates a request only; returned skip data is aligned elsewhere.
module vfu_skip_lane
(
    input  wire [3:0]  token_tile_i,
    input  wire        valid_i,
    input  wire [8:0]  feature_i,
    input  wire [12:0] skip_base_i,
    input  wire [2:0]  mt_i,
    input  wire [3:0]  vfu_op,

    output wire        skip_re_o,
    output wire [13:0] skip_raddr_o
);

localparam [3:0] VFU_OP_RQ_RES = 4'h1;

wire        is_skip_op;
wire [13:0] feature_offset;

assign is_skip_op = (vfu_op == VFU_OP_RQ_RES);

assign skip_re_o = valid_i && is_skip_op;

assign feature_offset = feature_i * mt_i;

assign skip_raddr_o = {1'b0, skip_base_i}
                   + feature_offset
                   + {10'b0, token_tile_i};

endmodule
