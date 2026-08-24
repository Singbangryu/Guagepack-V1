`timescale 1ns/1ps

// 16 independent S24 rowmax accumulators.
// One accepted beat carries one key column for all 16 query rows.
module vfu_rowmax16 (
    input  wire         clk_i,
    input  wire         rst_ni,
    input  wire         ce_i,

    input  wire         valid_i,
    input  wire         clear_i,
    input  wire         last_i,
    input  wire         key_valid_i,
    input  wire [383:0] score_i,          // 16 x S24

    output reg          result_valid_o,
    output reg  [383:0] rowmax_o          // 16 x S24
);

    localparam [23:0] S24_MIN = 24'h800000;
    integer lane;

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            result_valid_o <= 1'b0;
            rowmax_o       <= {16{S24_MIN}};
        end else if (ce_i) begin
            result_valid_o <= valid_i && last_i;

            if (valid_i) begin
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    if (clear_i) begin
                        rowmax_o[lane*24 +: 24]
                            <= key_valid_i
                             ? score_i[lane*24 +: 24]
                             : S24_MIN;
                    end else if (key_valid_i
                              && ($signed(score_i[lane*24 +: 24])
                                  > $signed(rowmax_o[lane*24 +: 24]))) begin
                        rowmax_o[lane*24 +: 24]
                            <= score_i[lane*24 +: 24];
                    end
                end
            end
        end
    end

endmodule
