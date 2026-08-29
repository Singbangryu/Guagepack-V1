`timescale 1ns/1ps

// Opaque 16-bank S-pad memory leaf.
//
// Each bank stores one 32-bit slice of the 512-bit vector at a common
// address.  Read latency counts CE-enabled pipeline advances.  The legal
// RD_LAT values are 1 and 2.
module vfu_spad_mem16 #(
    parameter DEPTH  = 512,
    parameter ADDR_W = 9,
    parameter RD_LAT = 2
) (
    input  wire                  clk_i,
    input  wire                  rst_ni,
    input  wire                  ce_i,

    input  wire                  wr_en_i,
    input  wire [ADDR_W-1:0]     wr_addr_i,
    input  wire [511:0]          wr_data_i,

    input  wire                  rd_en_i,
    input  wire [ADDR_W-1:0]     rd_addr_i,
    output wire                  rd_valid_o,
    output wire [511:0]          rd_data_o
);

    reg [31:0] mem_bank_00 [0:DEPTH-1];
    reg [31:0] mem_bank_01 [0:DEPTH-1];
    reg [31:0] mem_bank_02 [0:DEPTH-1];
    reg [31:0] mem_bank_03 [0:DEPTH-1];
    reg [31:0] mem_bank_04 [0:DEPTH-1];
    reg [31:0] mem_bank_05 [0:DEPTH-1];
    reg [31:0] mem_bank_06 [0:DEPTH-1];
    reg [31:0] mem_bank_07 [0:DEPTH-1];
    reg [31:0] mem_bank_08 [0:DEPTH-1];
    reg [31:0] mem_bank_09 [0:DEPTH-1];
    reg [31:0] mem_bank_10 [0:DEPTH-1];
    reg [31:0] mem_bank_11 [0:DEPTH-1];
    reg [31:0] mem_bank_12 [0:DEPTH-1];
    reg [31:0] mem_bank_13 [0:DEPTH-1];
    reg [31:0] mem_bank_14 [0:DEPTH-1];
    reg [31:0] mem_bank_15 [0:DEPTH-1];

    reg         rd_stage_valid_r;
    reg [511:0] rd_stage_data_r;

    // Reset intentionally excludes the memory and every data register.
    // It only cancels outstanding read-control state.  Reset also dominates
    // CE, so neither a read nor a write is accepted while reset is asserted.
    always @(posedge clk_i) begin
        if (!rst_ni) begin
            rd_stage_valid_r <= 1'b0;
        end else if (ce_i) begin
            rd_stage_valid_r <= rd_en_i;

            if (wr_en_i) begin
                mem_bank_00[wr_addr_i] <= wr_data_i[ 31:  0];
                mem_bank_01[wr_addr_i] <= wr_data_i[ 63: 32];
                mem_bank_02[wr_addr_i] <= wr_data_i[ 95: 64];
                mem_bank_03[wr_addr_i] <= wr_data_i[127: 96];
                mem_bank_04[wr_addr_i] <= wr_data_i[159:128];
                mem_bank_05[wr_addr_i] <= wr_data_i[191:160];
                mem_bank_06[wr_addr_i] <= wr_data_i[223:192];
                mem_bank_07[wr_addr_i] <= wr_data_i[255:224];
                mem_bank_08[wr_addr_i] <= wr_data_i[287:256];
                mem_bank_09[wr_addr_i] <= wr_data_i[319:288];
                mem_bank_10[wr_addr_i] <= wr_data_i[351:320];
                mem_bank_11[wr_addr_i] <= wr_data_i[383:352];
                mem_bank_12[wr_addr_i] <= wr_data_i[415:384];
                mem_bank_13[wr_addr_i] <= wr_data_i[447:416];
                mem_bank_14[wr_addr_i] <= wr_data_i[479:448];
                mem_bank_15[wr_addr_i] <= wr_data_i[511:480];
            end

            if (rd_en_i) begin
                rd_stage_data_r[ 31:  0] <= mem_bank_00[rd_addr_i];
                rd_stage_data_r[ 63: 32] <= mem_bank_01[rd_addr_i];
                rd_stage_data_r[ 95: 64] <= mem_bank_02[rd_addr_i];
                rd_stage_data_r[127: 96] <= mem_bank_03[rd_addr_i];
                rd_stage_data_r[159:128] <= mem_bank_04[rd_addr_i];
                rd_stage_data_r[191:160] <= mem_bank_05[rd_addr_i];
                rd_stage_data_r[223:192] <= mem_bank_06[rd_addr_i];
                rd_stage_data_r[255:224] <= mem_bank_07[rd_addr_i];
                rd_stage_data_r[287:256] <= mem_bank_08[rd_addr_i];
                rd_stage_data_r[319:288] <= mem_bank_09[rd_addr_i];
                rd_stage_data_r[351:320] <= mem_bank_10[rd_addr_i];
                rd_stage_data_r[383:352] <= mem_bank_11[rd_addr_i];
                rd_stage_data_r[415:384] <= mem_bank_12[rd_addr_i];
                rd_stage_data_r[447:416] <= mem_bank_13[rd_addr_i];
                rd_stage_data_r[479:448] <= mem_bank_14[rd_addr_i];
                rd_stage_data_r[511:480] <= mem_bank_15[rd_addr_i];
            end
        end
    end

    generate
        if (RD_LAT == 1) begin : g_rd_lat_1
            assign rd_valid_o = rd_stage_valid_r;
            assign rd_data_o  = rd_stage_data_r;
        end else begin : g_rd_lat_2
            reg         rd_out_valid_r;
            reg [511:0] rd_out_data_r;

            always @(posedge clk_i) begin
                if (!rst_ni) begin
                    rd_out_valid_r <= 1'b0;
                end else if (ce_i) begin
                    rd_out_valid_r <= rd_stage_valid_r;
                    if (rd_stage_valid_r)
                        rd_out_data_r <= rd_stage_data_r;
                end
            end

            assign rd_valid_o = rd_out_valid_r;
            assign rd_data_o  = rd_out_data_r;
        end
    endgenerate

endmodule
