`timescale 1ns/1ps

// =============================================================================
// GaugePack VFU signed RNE right shifter
// =============================================================================
// Pure combinational S3 helper.
//
// Function: y_o = round_to_nearest_even(x_i / 2^shamt_i)
//
// x_i is a signed two's-complement S48 value. The arithmetic right shift
// first produces the floor value. Adding increment_w therefore implements RNE
// for both positive and negative inputs without taking an absolute value.
//
//     increment = guard & (sticky | shifted_lsb)
//
// =============================================================================

module vfu_rne_shift48 (
    input  wire signed [47:0] x_i,
    input  wire        [5:0]  shamt_i,
    output wire signed [47:0] y_o
);

    wire               shift_valid_w;
    wire        [5:0]  shamt_m1_w;
    wire        [47:0] guard_mask_w;
    wire        [47:0] sticky_mask_w;
    wire signed [47:0] shifted_w;
    wire               guardbit_w;
    wire               sticky_bit_w;
    wire               increment_w;
    wire signed [47:0] rounded_w;

    assign shift_valid_w = (shamt_i != 6'd0) && (shamt_i < 6'd48);
    assign shamt_m1_w     = shift_valid_w ? (shamt_i - 6'd1) : 6'd0;

    // Example: shamt=4
    // guard_mask  = 1 << 3 = 4'b1000, selecting x_i[3].
    // sticky_mask = guard_mask - 1 = 3'b111, selecting x_i[2:0].
    assign guard_mask_w  = shift_valid_w ? (48'd1 << shamt_m1_w) : 48'd0;
    assign sticky_mask_w = shift_valid_w ? (guard_mask_w - 48'd1) : 48'd0;

    assign shifted_w    = $signed(x_i) >>> shamt_i;
    assign guardbit_w   = |(x_i & guard_mask_w);
    assign sticky_bit_w = |(x_i & sticky_mask_w);
    assign increment_w  = guardbit_w & (shifted_w[0] | sticky_bit_w);
    assign rounded_w    = shifted_w + {{47{1'b0}}, increment_w};

    assign y_o = (shamt_i < 6'd48) ? rounded_w : 48'sd0;

endmodule
