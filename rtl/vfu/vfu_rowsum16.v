`timescale 1ns/1ps

// 16 independent U13 rowsum accumulators.
// E is U7 stored in an 8-bit physical byte with bit 7 equal to zero.
module vfu_rowsum16 (
    input  wire         clk_i,
    input  wire         rst_ni,
    input  wire         ce_i,

    input  wire         valid_i,
    input  wire         clear_i,
    input  wire         last_i,
    input  wire         key_valid_i,
    input  wire [127:0] e_i,              // 16 x {1'b0, U7}

    output reg          result_valid_o,
    output reg  [207:0] rowsum_o          // 16 x U13
);

    integer lane;

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            result_valid_o <= 1'b0;
            rowsum_o       <= 208'd0;
        end else if (ce_i) begin
            result_valid_o <= valid_i && last_i;

            if (valid_i) begin
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    if (clear_i) begin
                        rowsum_o[lane*13 +: 13]
                            <= key_valid_i
                             ? {6'd0, e_i[lane*8 +: 7]}
                             : 13'd0;
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
