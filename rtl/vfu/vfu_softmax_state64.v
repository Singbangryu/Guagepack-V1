`timescale 1ns/1ps

// One-head Softmax row-state storage.
// Four query tiles each hold sixteen 24-bit rows.  A tile is written
// atomically and is tagged as either MAX (type 0) or reciprocal R (type 1).
module vfu_softmax_state64 (
    input  wire         clk_i,
    input  wire         rst_ni,
    input  wire         ce_i,
    input  wire         clear_i,

    input  wire         wr_en_i,
    input  wire [1:0]   wr_tile_i,
    input  wire         wr_type_i,
    input  wire [383:0] wr_data_i,

    input  wire [1:0]   rd_tile_i,
    output reg          rd_valid_o,
    output reg          rd_type_o,
    output reg  [383:0] rd_data_o
);

    // Explicit tile registers keep this leaf as fixed FF state rather than
    // exposing a configurable memory interface.  Data is intentionally not
    // reset or cleared; invalid metadata masks it from the read port.
    reg [383:0] tile0_data_q;
    reg [383:0] tile1_data_q;
    reg [383:0] tile2_data_q;
    reg [383:0] tile3_data_q;
    reg [3:0]   tile_valid_q;
    reg [3:0]   tile_type_q;

    // Masked asynchronous read.  There is intentionally no write bypass.
    always @* begin
        rd_valid_o = 1'b0;
        rd_type_o  = 1'b0;
        rd_data_o  = 384'd0;

        case (rd_tile_i)
            2'd0: begin
                if (tile_valid_q[0]) begin
                    rd_valid_o = 1'b1;
                    rd_type_o  = tile_type_q[0];
                    rd_data_o  = tile0_data_q;
                end
            end
            2'd1: begin
                if (tile_valid_q[1]) begin
                    rd_valid_o = 1'b1;
                    rd_type_o  = tile_type_q[1];
                    rd_data_o  = tile1_data_q;
                end
            end
            2'd2: begin
                if (tile_valid_q[2]) begin
                    rd_valid_o = 1'b1;
                    rd_type_o  = tile_type_q[2];
                    rd_data_o  = tile2_data_q;
                end
            end
            2'd3: begin
                if (tile_valid_q[3]) begin
                    rd_valid_o = 1'b1;
                    rd_type_o  = tile_type_q[3];
                    rd_data_o  = tile3_data_q;
                end
            end
        endcase
    end

    // Synchronous priority: reset, CE hold, clear, then atomic tile write.
    always @(posedge clk_i) begin
        if (!rst_ni) begin
            tile_valid_q <= 4'b0000;
            tile_type_q  <= 4'b0000;
        end else if (ce_i) begin
            if (clear_i) begin
                tile_valid_q <= 4'b0000;
                tile_type_q  <= 4'b0000;
            end else if (wr_en_i) begin
                case (wr_tile_i)
                    2'd0: tile0_data_q <= wr_data_i;
                    2'd1: tile1_data_q <= wr_data_i;
                    2'd2: tile2_data_q <= wr_data_i;
                    2'd3: tile3_data_q <= wr_data_i;
                endcase
                tile_valid_q[wr_tile_i] <= 1'b1;
                tile_type_q[wr_tile_i]  <= wr_type_i;
            end
        end
    end

endmodule
