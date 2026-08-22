`timescale 1ns/1ps

// =============================================================================
// GaugePack VFU exact residual adder
// =============================================================================
// Both inputs are already quantized into the same native S8 tensor scale.
// Architectural signed activation code -128 is forbidden, so each input is
// in [-127, 127]. The exact result is therefore S9 [-254, 254].
//
// No rounding or post-add saturation is permitted. The S9 result feeds the
// following LayerNorm statistics path directly.
// =============================================================================

module vfu_residual_add (
    input  wire signed [7:0] main_i,
    input  wire signed [7:0] skip_i,
    output wire signed [8:0] sum_o
);

    wire signed [8:0] main_ext_w;
    wire signed [8:0] skip_ext_w;

    assign main_ext_w = {main_i[7], main_i};
    assign skip_ext_w = {skip_i[7], skip_i};
    assign sum_o      = main_ext_w + skip_ext_w;

endmodule
