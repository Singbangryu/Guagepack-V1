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
// Boundary behavior:
//   shamt == 0  : exact bypass
//   1..47       : normal signed RNE
//   shamt >= 48 : result is zero; every S48 input has magnitude <= 0.5 LSB
//                 at that output scale, and the sole exact tie (-2^47/2^48)
//                 rounds to the even integer zero.
//
// No saturation or output narrowing is performed here. Those operations
// belong to the following operator-specific S3 formatting logic.
// =============================================================================

module vfu_rne_shift48 (
    input  wire signed [47:0] x_i,
    input  wire        [5:0]  shamt_i,
    output wire signed [47:0] y_o
);

    reg        [47:0] sticky_mask_w;
    reg signed [47:0] shifted_w;
    reg               guardbit_w;
    reg               shifted_lsb_w;
    reg               increment_w;
    reg               sticky_bit_w;
    reg signed [47:0] rne_y;

    always @(*) begin
        shifted_w     = 48'sd0;
        sticky_mask_w = 48'd0;
        guardbit_w    = 1'b0;
        shifted_lsb_w = 1'b0;
        increment_w   = 1'b0;
        sticky_bit_w  = 1'b0;
        rne_y          = 48'sd0;

        if (shamt_i == 6'd0) begin
            rne_y = x_i;
        end else if (shamt_i < 6'd48) begin
            shifted_w = $signed(x_i) >>> shamt_i;

            guardbit_w = x_i[shamt_i - 6'd1];

            // Example: shamt=4
            // (1 << 3) - 1 = 3'b111, selecting x_i[2:0].
            sticky_mask_w = (48'd1 << (shamt_i - 6'd1)) - 48'd1;
            sticky_bit_w  = |(x_i & sticky_mask_w);

            shifted_lsb_w = shifted_w[0];
            increment_w   = guardbit_w & (shifted_lsb_w | sticky_bit_w);
            rne_y          = shifted_w + {{47{1'b0}}, increment_w};
        end
    end

    assign y_o = rne_y;

endmodule
