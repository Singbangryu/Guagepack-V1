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
// Therefore this block owns 16 independent row states.  It does NOT reduce the
// 16 lanes into one scalar.  On every accepted key beat it updates:
//
//   rowmax[q] = max(rowmax[q], score[q,key])
//   rowsum[q] = rowsum[q] + E[q,key]
//
// while score_accept_o/e_accept_o qualify the simultaneous external scratch
// write.  Scratch storage itself remains outside this block, so the score/E
// row is not duplicated in registers here.
//
// clear_i marks key beat 0 of a 16-query group.  last_i marks the final key
// beat.  Lifecycle sidebands are consumed only with an accepted valid beat.
// done_o and the final packed state become visible on that same accepted last
// edge.  A global ce_i stall holds both state and done_o.
// =============================================================================

module vfu_softmax (
    input  wire                     clk_i,
    input  wire                     rst_ni,
    input  wire                     ce_i,

    // -------------------------------------------------------------------------
    // QK score ingest / rowmax phase: 16 query rows at one key address
    // -------------------------------------------------------------------------
    input  wire                     rowmax_valid_i,
    input  wire                     rowmax_clear_i,
    input  wire                     rowmax_last_i,
    input  wire [15:0]              score_lane_valid_i,
    input  wire [383:0]             score_i,             // 16 x S24

    // Use this as the score-scratch write qualifier.  score_i and
    // score_lane_valid_i are the accompanying payload.
    output wire                     score_accept_o,

    // 16 x S24 row maxima and one validity bit per query row.  An all-masked
    // query row returns rowmax=0 and has_valid=0.
    output reg                      rowmax_done_o,
    output reg  [383:0]             rowmax_o,
    output reg  [15:0]              rowmax_has_valid_o,

    // -------------------------------------------------------------------------
    // QEXP output ingest / rowsum phase: 16 query rows at one key address
    // -------------------------------------------------------------------------
    input  wire                     rowsum_valid_i,
    input  wire                     rowsum_clear_i,
    input  wire                     rowsum_last_i,
    input  wire [15:0]              e_lane_valid_i,
    input  wire [127:0]             e_i,                 // 16 x E7 byte

    // Use this as the E-scratch write qualifier.  e_i and e_lane_valid_i are
    // the accompanying payload.
    output wire                     e_accept_o,

    // 16 x U13 rowsums.  rowsum_zero_o directly supplies the per-query L==0
    // semantic sideband for the reciprocal phase.
    output reg                      rowsum_done_o,
    output reg  [207:0]             rowsum_o,
    output wire [15:0]              rowsum_zero_o
);

    integer lane;

    assign score_accept_o = rst_ni && ce_i && rowmax_valid_i;
    assign e_accept_o     = rst_ni && ce_i && rowsum_valid_i;

    genvar zero_lane;
    generate
        for (zero_lane = 0; zero_lane < 16; zero_lane = zero_lane + 1) begin : GEN_ROWSUM_ZERO
            assign rowsum_zero_o[zero_lane] =
                (rowsum_o[zero_lane*13 +: 13] == 13'd0);
        end
    endgenerate

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            rowmax_done_o      <= 1'b0;
            rowmax_o           <= 384'd0;
            rowmax_has_valid_o <= 16'd0;

            rowsum_done_o      <= 1'b0;
            rowsum_o           <= 208'd0;
        end else if (ce_i) begin
            // -----------------------------------------------------------------
            // Sixteen independent rowmax accumulators.
            // -----------------------------------------------------------------
            rowmax_done_o <= rowmax_valid_i && rowmax_last_i;

            if (rowmax_valid_i) begin
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    if (rowmax_clear_i) begin
                        if (score_lane_valid_i[lane]) begin
                            rowmax_o[lane*24 +: 24]
                                <= score_i[lane*24 +: 24];
                            rowmax_has_valid_o[lane] <= 1'b1;
                        end else begin
                            rowmax_o[lane*24 +: 24] <= 24'd0;
                            rowmax_has_valid_o[lane] <= 1'b0;
                        end
                    end else if (score_lane_valid_i[lane]) begin
                        if (!rowmax_has_valid_o[lane]
                            || ($signed(score_i[lane*24 +: 24])
                                > $signed(rowmax_o[lane*24 +: 24]))) begin
                            rowmax_o[lane*24 +: 24]
                                <= score_i[lane*24 +: 24];
                        end

                        rowmax_has_valid_o[lane] <= 1'b1;
                    end
                end
            end

            // -----------------------------------------------------------------
            // Sixteen independent U13 rowsum accumulators.
            // -----------------------------------------------------------------
            rowsum_done_o <= rowsum_valid_i && rowsum_last_i;

            if (rowsum_valid_i) begin
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    if (rowsum_clear_i) begin
                        rowsum_o[lane*13 +: 13]
                            <= e_lane_valid_i[lane]
                             ? {6'd0, e_i[lane*8 +: 7]}
                             : 13'd0;
                    end else if (e_lane_valid_i[lane]) begin
                        rowsum_o[lane*13 +: 13]
                            <= rowsum_o[lane*13 +: 13]
                             + {6'd0, e_i[lane*8 +: 7]};
                    end
                end
            end
        end
    end

endmodule
