`timescale 1ns/1ps
// DRAFT: single-current-tile LN rho feedback, 16 U8 values plus valid/tag.
// Capture must already mean selected LN + real S3 retirement + LN_RSQRT,
// with result-aligned tile. Clear is LN init only, never Softmax row-state init.
// Pack each S32 container's LOW BYTE; taking capture_data[127:0] is incorrect.
module vfu_ln_rho16_draft (
    input  wire         clk_i,
    input  wire         rst_ni,
    input  wire         clear_i,
    input  wire         capture_i,
    input  wire [1:0]   capture_tile_i,
    input  wire [511:0] capture_data_i,
    input  wire [1:0]   rd_tile_i,
    output wire [127:0] rho_o,
    output wire         match_valid_o,
    output reg          stored_o
);
    reg [127:0] rho_r;
    reg [1:0] tile_r;
    reg valid_r;
    integer lane;
    assign match_valid_o = valid_r && tile_r == rd_tile_i;
    assign rho_o = match_valid_o ? rho_r : 128'd0;
    always @(posedge clk_i) begin
        if (!rst_ni || clear_i) begin
            valid_r <= 1'b0;
            tile_r <= 2'd0;
            stored_o <= 1'b0;
        end else begin
            stored_o <= capture_i;
            if (capture_i) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    rho_r[lane*8 +: 8] <= capture_data_i[lane*32 +: 8];
                tile_r <= capture_tile_i;
                valid_r <= 1'b1;
            end
        end
    end
    // synthesis translate_off
    integer check_lane;
    always @(posedge clk_i) begin
        if (rst_ni && clear_i && capture_i)
            $fatal(1, "LN rho clear and capture conflict");
        if (rst_ni && !clear_i && capture_i) begin
            for (check_lane = 0; check_lane < 16; check_lane = check_lane + 1)
                if (capture_data_i[check_lane*32+8 +: 24] !== 24'd0)
                    $fatal(1, "LN rho capture requires zero-extended U8 containers");
        end
    end
    // synthesis translate_on
endmodule
