`timescale 1ns/1ps

// 16-lane QEXP E commit + 16 independent U13 rowsum accumulators.
// e_i is already POST-ALU-clamped E7 and is stored in an 8-bit physical
// byte as {1'b0, E[6:0]}. Invalid padded keys commit E=0.
module vfu_exp_commit16 (
    input  wire         clk_i,
    input  wire         rst_ni,
    input  wire         ce_i,

    input  wire         valid_i,
    input  wire         clear_i,
    input  wire         last_i,
    input  wire         key_valid_i,

    input  wire [127:0] e_i,              // 16 x physical E7 byte

    output reg          result_valid_o,   // Final rowsum valid
    output wire [127:0] e_o,              // Key-masked E scratch write data
    output reg  [207:0] rowsum_o          // 16 x U13; max = 127 x 64 = 8128
);

    wire [127:0] e_commit;

    // Invalid padded key must still commit E=0 to its scheduled address.
    assign e_commit = key_valid_i ? e_i : 128'd0;
    assign e_o      = e_commit;

    integer lane;

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            result_valid_o <= 1'b0;
            rowsum_o       <= 208'd0;
        end else if (ce_i) begin
            // key_valid_i gates arithmetic only; the scheduled last beat
            // still produces result_valid_o even when it is padding.
            result_valid_o <= valid_i && last_i;

            if (valid_i) begin
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    if (clear_i) begin
                        rowsum_o[lane*13 +: 13]
                            <= {6'd0, e_commit[lane*8 +: 7]};
                    end else if (key_valid_i) begin
                        rowsum_o[lane*13 +: 13]
                            <= rowsum_o[lane*13 +: 13]
                             + {6'd0, e_i[lane*8 +: 7]};
                    end
                end
            end
        end
    end

endmodule
