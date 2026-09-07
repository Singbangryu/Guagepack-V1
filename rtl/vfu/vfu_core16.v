`timescale 1ns/1ps
`include "vfu_internal_op_defs.vh"

module vfu_core16 #(
    parameter PAGE_ID_W = 8
) (
    input  wire                    clk_i,
    input  wire                    rst_ni,
    input  wire                    valid_i,
    input  wire [3:0]              op_i,
    input  wire [8:0]              feature_i,
    input  wire [3:0]              token_tile_i,
    input  wire [15:0]             lane_valid_i,
    input  wire                    key_valid_i,
    input  wire                    last_i,

    input  wire [511:0]            src0_i,
    input  wire [511:0]            src1_i,
    input  wire [127:0]            rsqrted_i,
    input  wire [PAGE_ID_W-1:0]     page_id_i,

    input  wire [127:0]            skip_s1_i,

    output wire                    s0_valid_o,
    output wire [3:0]              s0_op_o,
    output wire [8:0]              s0_feature_o,
    output wire [3:0]              s0_token_tile_o,

    output wire                    valid_o,
    output wire [3:0]              op_o,
    output wire [8:0]              feature_o,
    output wire [3:0]              token_tile_o,
    output wire [15:0]             lane_valid_o,
    output wire                    key_valid_o,
    output wire                    last_o,
    output wire [511:0]            data_o,

    output wire                    moment_capture_o,
    output wire [255:0]            moment_s_o,
    output wire [367:0]            moment_q_o
);

    wire [404:0] boundary_flat;
    wire signed [26:0] x_min, x_max;
    wire [7:0] low_code, high_code;
    wire [287:0] coeff_m_s0;
    wire [767:0] coeff_c_s0;
    wire [95:0] coeff_shamt_s0;

    wire valid_s0;
    wire [3:0] op_s0;
    wire [8:0] feature_s0;
    wire [15:0] lane_valid_s0;
    wire [431:0] a_s0;
    wire [287:0] b_s0;
    wire [767:0] c_s0;
    wire [63:0] seg_addr_s0;
    wire [31:0] range_s0;
    reg [3:0] token_tile_s0_r;
    reg key_valid_s0_r, last_s0_r;

    vfu_coeff_page16 #(.PAGE_ID_W(PAGE_ID_W)) u_coeff_page (
        .page_id_i       (page_id_i),
        .op_s0_i         (op_s0),
        .feature_s0_i    (feature_s0),
        .seg_addr_s0_i   (seg_addr_s0),
        .boundary_flat_o (boundary_flat),
        .x_min_o         (x_min),
        .x_max_o         (x_max),
        .low_code_o      (low_code),
        .high_code_o     (high_code),
        .m_o             (coeff_m_s0),
        .c_o             (coeff_c_s0),
        .shamt_o         (coeff_shamt_s0)
    );

    vfu_pre_alu16 u_pre (
        .clk_i           (clk_i),
        .rst_ni          (rst_ni),
        .valid_i         (valid_i),
        .op_i            (op_i),
        .feature_i       (feature_i),
        .lane_valid_i    (lane_valid_i),
        .src0_i          (src0_i),
        .src1_i          (src1_i),
        .rsqrted_i       (rsqrted_i),
        .boundary_flat_i (boundary_flat),
        .x_min_i         (x_min),
        .x_max_i         (x_max),
        .valid_o         (valid_s0),
        .op_o            (op_s0),
        .feature_o       (feature_s0),
        .lane_valid_o    (lane_valid_s0),
        .a_o             (a_s0),
        .b_o             (b_s0),
        .c_o             (c_s0),
        .seg_addr_o      (seg_addr_s0),
        .seg_range_o     (range_s0)
    );

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            token_tile_s0_r <= 4'd0;
            key_valid_s0_r  <= 1'b0;
            last_s0_r       <= 1'b0;
        end else if (valid_i) begin
            token_tile_s0_r <= token_tile_i;
            key_valid_s0_r  <= key_valid_i;
            last_s0_r       <= last_i;
        end
    end

    assign s0_valid_o      = valid_s0;
    assign s0_op_o         = op_s0;
    assign s0_feature_o    = feature_s0;
    assign s0_token_tile_o = token_tile_s0_r;

    reg [287:0] dsp_b;
    reg [767:0] dsp_c;
    reg [95:0] shamt_s0;

    always @(*) begin
        dsp_b    = b_s0;
        dsp_c    = c_s0;
        shamt_s0 = 96'd0;
        case (op_s0)
            `VFU_OP_RQ, `VFU_OP_RQ_RES, `VFU_OP_GELU,
            `VFU_OP_QEXP, `VFU_OP_LN_RSQRT, `VFU_OP_LN_AFFINE: begin
                dsp_b    = coeff_m_s0;
                dsp_c    = coeff_c_s0;
                shamt_s0 = coeff_shamt_s0;
            end
            `VFU_OP_SM_RECIP_RAW: begin
                dsp_b    = coeff_m_s0;
                dsp_c    = coeff_c_s0;
                shamt_s0 = {16{6'd8}};
            end
            `VFU_OP_SM_CONTEXT: shamt_s0 = {16{6'd23}};
            `VFU_OP_LN_D:       shamt_s0 = {16{6'd4}};
            default: begin end
        endcase
    end

    reg valid_s1_r;

    reg [30:0] meta_s1_r, meta_s2_r;
    reg [95:0] shamt_s1_r, shamt_s2_r;
    reg [31:0] range_s1_r, range_s2_r;
    reg [15:0] tails_s1_r, tails_s2_r;
    reg [127:0] skip_s2_r;

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            valid_s1_r <= 1'b0;
            meta_s1_r  <= 31'd0;
            meta_s2_r  <= 31'd0;
            shamt_s1_r <= 96'd0;
            shamt_s2_r <= 96'd0;
            range_s1_r <= 32'd0;
            range_s2_r <= 32'd0;
            tails_s1_r <= 16'd0;
            tails_s2_r <= 16'd0;
            skip_s2_r  <= 128'd0;
        end else begin
            valid_s1_r <= valid_s0;
            if (valid_s0) begin
                meta_s1_r <= {feature_s0, token_tile_s0_r, lane_valid_s0,
                              key_valid_s0_r, last_s0_r};
                shamt_s1_r <= shamt_s0;
                range_s1_r <= range_s0;
                tails_s1_r <= {low_code, high_code};
            end
            if (valid_s1_r) begin
                meta_s2_r  <= meta_s1_r;
                shamt_s2_r <= shamt_s1_r;
                range_s2_r <= range_s1_r;
                tails_s2_r <= tails_s1_r;
                skip_s2_r  <= skip_s1_i;
            end
        end
    end

    wire valid_s2;
    wire [3:0] op_s2;
    wire [767:0] p_s2;
    wire [8:0] feature_s2;
    wire [3:0] token_tile_s2;
    wire [15:0] lane_valid_s2;
    wire key_valid_s2, last_s2;

    assign {feature_s2, token_tile_s2, lane_valid_s2, key_valid_s2, last_s2}
        = meta_s2_r;

    vfu_dsp16 u_dsp (
        .clk_i   (clk_i),
        .rst_ni  (rst_ni),
        .valid_i (valid_s0),
        .op_i    (op_s0),
        .a_i     (a_s0),
        .b_i     (dsp_b),
        .c_i     (dsp_c),
        .valid_o (valid_s2),
        .op_o    (op_s2),
        .p_o     (p_s2)
    );

    vfu_post_alu16_s3 u_post (
        .clk_i            (clk_i),
        .rst_ni           (rst_ni),
        .valid_i          (valid_s2),
        .op_i             (op_s2),
        .feature_i        (feature_s2),
        .token_tile_i     (token_tile_s2),
        .lane_valid_i     (lane_valid_s2),
        .key_valid_i      (key_valid_s2),
        .last_i           (last_s2),
        .p_i              (p_s2),
        .shamt_i          (shamt_s2_r),
        .range_i          (range_s2_r),
        .low_code_i       (tails_s2_r[15:8]),
        .high_code_i      (tails_s2_r[7:0]),
        .skip_i           (skip_s2_r),
        .valid_o          (valid_o),
        .op_o             (op_o),
        .feature_o        (feature_o),
        .token_tile_o     (token_tile_o),
        .lane_valid_o     (lane_valid_o),
        .key_valid_o      (key_valid_o),
        .last_o           (last_o),
        .data_o           (data_o),
        .moment_capture_o (moment_capture_o),
        .moment_s_o       (moment_s_o),
        .moment_q_o       (moment_q_o)
    );
endmodule
