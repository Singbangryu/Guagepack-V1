`timescale 1ns/1ps
`include "vfu_defs.vh"

// =============================================================================
// GaugePack VFU activation clamp wrapper
// =============================================================================
// One shared clamp datapath per lane. The caller supplies only the clamp mode;
// operation decoding remains outside this reusable arithmetic block.
//
// Modes:
//   VFU_CLAMP_S8_SYM : [-127, 127], output byte is signed two's-complement
//   VFU_CLAMP_U7     : [   0, 127], output byte is unsigned
//   VFU_CLAMP_U8     : [   0, 255], output byte is unsigned
//
// Mode 2'b11 is reserved and produces zero.
// =============================================================================

module vfu_clamp_wrapper (
    input  wire signed [47:0] x_i,
    input  wire        [1:0]  clamp_mode_i,
    output wire        [7:0]  y_o
);

    wire mode_s8_w;
    wire mode_u7_w;
    wire mode_u8_w;
    wire mode_valid_w;

    wire signed [47:0] lower_bound_w;
    wire signed [47:0] upper_bound_w;
    wire        [7:0]  lower_code_w;
    wire        [7:0]  upper_code_w;

    assign mode_s8_w   = (clamp_mode_i == `VFU_CLAMP_S8_SYM);
    assign mode_u7_w   = (clamp_mode_i == `VFU_CLAMP_U7);
    assign mode_u8_w   = (clamp_mode_i == `VFU_CLAMP_U8);
    assign mode_valid_w = mode_s8_w | mode_u7_w | mode_u8_w;

    // S8 differs at the lower bound; U8 differs at the upper bound.
    // Selecting bounds before comparison avoids duplicating full clamp paths.
    assign lower_bound_w = mode_s8_w ? -48'sd127 : 48'sd0;
    assign upper_bound_w = mode_u8_w ?  48'sd255 : 48'sd127;
    assign lower_code_w  = mode_s8_w ?  8'h81     : 8'h00;
    assign upper_code_w  = mode_u8_w ?  8'hff     : 8'h7f;

    assign y_o = !mode_valid_w       ? 8'h00 :
                 (x_i < lower_bound_w) ? lower_code_w :
                 (x_i > upper_bound_w) ? upper_code_w :
                                          x_i[7:0];

endmodule
