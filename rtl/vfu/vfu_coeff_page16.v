`timescale 1ns/1ps

module vfu_coeff_page16 #(
    parameter PAGE_ID_W = 8
) (
    input  wire [PAGE_ID_W-1:0] page_id_i,
    input  wire [3:0]          op_s0_i,
    input  wire [8:0]          feature_s0_i,
    input  wire [63:0]         seg_addr_s0_i,
    output wire [404:0]        boundary_flat_o,
    output wire signed [26:0]  x_min_o,
    output wire signed [26:0]  x_max_o,
    output wire [7:0]          low_code_o,
    output wire [7:0]          high_code_o,
    output wire [287:0]        m_o,
    output wire [767:0]        c_o,
    output wire [95:0]         shamt_o
);

endmodule
