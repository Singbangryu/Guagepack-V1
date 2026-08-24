`timescale 1ns/1ps

// =============================================================================
// GaugePack Softmax row state -- 16 query lanes
// =============================================================================
// Frozen scratch orientation:
//
//   score scratch address = key index 0..63
//   score_i[24*q +: 24]   = signed score[q,key], q=0..15
//   e_i[8*q +: 8]         = E7[q,key] in a physical byte, bit7=0
//
// One beat contains 16 query rows for one common key.  GaugePack v1 does not
// mask query lanes; key_valid_i is the scalar validity of that key and is
// shared by all 16 rows.
//
// Rowmax ignores an invalid key.  QEXP must turn an invalid key into E=0 before
// its output reaches this block, so rowsum needs no second mask input.
//
// Every legal Softmax command contains at least one valid key.  Therefore each
// row has a maximum, its d=0 QEXP term is 127, and 127 <= rowsum <= 8128.
// L=0 is not an architectural state.
//
// Scratch storage itself remains outside this block.  clear_i marks the first
// key beat of a 16-query group, and last_i marks the final key beat.  Inputs are
// consumed only when ce_i and the matching rowmax_valid_i/rowsum_valid_i input
// is high.  A ce_i stall holds state and done.
// =============================================================================

module vfu_softmax (
    input  wire                     clk_i,
    input  wire                     rst_ni,
    input  wire                     ce_i,

    // Scalar mask for the current key; common to all 16 query rows.
    input  wire                     key_valid_i,

    // -------------------------------------------------------------------------
    // QK score ingest / rowmax phase: 16 query rows at one key address
    // -------------------------------------------------------------------------
    input  wire                     rowmax_valid_i,
    input  wire                     rowmax_clear_i,
    input  wire                     rowmax_last_i,
    input  wire [383:0]             score_i,             // 16 x S24

    output reg                      rowmax_done_o,
    output reg  [383:0]             rowmax_o,             // 16 x S24

    // -------------------------------------------------------------------------
    // QEXP output ingest / rowsum phase: 16 query rows at one key address
    // -------------------------------------------------------------------------
    input  wire                     rowsum_valid_i,
    input  wire                     rowsum_clear_i,
    input  wire                     rowsum_last_i,
    input  wire [127:0]             e_i,                 // 16 x E7 byte

    output reg                      rowsum_done_o,
    output reg  [207:0]             rowsum_o             // 16 x U13
);

    integer lane;

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            rowmax_done_o <= 1'b0;
            rowmax_o      <= 384'd0;

            rowsum_done_o <= 1'b0;
            rowsum_o      <= 208'd0;
        end else if (ce_i) begin
            // -----------------------------------------------------------------
            // Sixteen rowmax accumulators, all gated by one common key mask.
            // S24_MIN is only the initialization sentinel.  A legal command
            // always replaces/compares it with at least one valid key score.
            // -----------------------------------------------------------------
            rowmax_done_o <= rowmax_valid_i && rowmax_last_i;

            if (rowmax_valid_i) begin
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    if (rowmax_clear_i) begin
                        rowmax_o[lane*24 +: 24]
                            <= key_valid_i
                             ? score_i[lane*24 +: 24]
                             : 24'h800000;
                    end else if (key_valid_i
                              && ($signed(score_i[lane*24 +: 24])
                                  > $signed(rowmax_o[lane*24 +: 24]))) begin
                        rowmax_o[lane*24 +: 24]
                            <= score_i[lane*24 +: 24];
                    end
                end
            end

            // -----------------------------------------------------------------
            // Sixteen independent U13 rowsum accumulators.
            // Invalid keys already arrive as E=0 from QEXP POST.
            // -----------------------------------------------------------------
            rowsum_done_o <= rowsum_valid_i && rowsum_last_i;

            if (rowsum_valid_i) begin
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    if (rowsum_clear_i) begin
                        rowsum_o[lane*13 +: 13]
                            <= {6'd0, e_i[lane*8 +: 7]};
                    end else begin
                        rowsum_o[lane*13 +: 13]
                            <= rowsum_o[lane*13 +: 13]
                             + {6'd0, e_i[lane*8 +: 7]};
                    end
                end
            end
        end
    end

endmodule
