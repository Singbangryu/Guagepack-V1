`timescale 1ns/1ps
`include "vfu_internal_op_defs.vh"

// Arithmetic-only draft; no memory, address generator, or command controller.
// Stage modules: vfu_pre_alu16, vfu_dsp16, vfu_post_alu16_s3.
// Stage modules own all arithmetic registers. CORE16 owns sideband alignment.
// Keep one op/phase until drain, except the Moment INIT -> ACC transition.
// page_id_i stays stable until drain. PAGE_ID_W is an opaque draft parameter.
// The controller supplies INIT on the first accepted Moment beat, then ACC;
// last_i marks the final beat of that reduction. Do not interleave reductions.
module vfu_core16_draft #(
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

    // RD_LAT=1 memory output, available alongside the DSP's S1 item.
    // E0: S0 loaded -> E1: memory response -> E2: skip_s2_q -> E3: POST.
    input  wire [127:0]            skip_s1_i,

    // Tap for an external skip request helper; no extra registers here.
    output wire                    s0_valid_o,
    output wire [3:0]              s0_op_o,
    output wire [8:0]              s0_feature_o,
    output wire [3:0]              s0_token_tile_o,

    // Registered S3 result. Each lane keeps its operation-specific S32/U32
    // container. lane_valid_o is metadata; final byte masking is downstream.
    output wire                    valid_o,
    output wire [3:0]              op_o,
    output wire [8:0]              feature_o,
    output wire [3:0]              token_tile_o,
    output wire [15:0]             lane_valid_o,
    output wire                    key_valid_o,
    output wire                    last_o,
    output wire [511:0]            data_o,

    // Final Moment fields only; no persistent per-tile state bank here.
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
    reg [3:0] token_tile_s0_q;
    reg key_valid_s0_q, last_s0_q;

    // Page headers feed PRE combinationally before the S0 edge.
    // M/C/shamt lookup uses registered S0 feature/segment addresses.
    vfu_coeff_page16_draft #(.PAGE_ID_W(PAGE_ID_W)) u_coeff_page (
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
        .ce_i            (1'b1),
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
            token_tile_s0_q <= 4'd0;
            key_valid_s0_q  <= 1'b0;
            last_s0_q       <= 1'b0;
        end else if (valid_i) begin
            token_tile_s0_q <= token_tile_i;
            key_valid_s0_q  <= key_valid_i;
            last_s0_q      <= last_i;
        end
    end

    assign s0_valid_o      = valid_s0;
    assign s0_op_o         = op_s0;
    assign s0_feature_o    = feature_s0;
    assign s0_token_tile_o = token_tile_s0_q;

    // S1 operand selection is COMBINATIONAL. The DSP owns the S1 registers.
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
            default: begin end // Moment and LN_NORM keep PRE operands/raw P.
        endcase
    end

    reg valid_s1_q;
    // {feature[8:0], tile[3:0], lane_valid[15:0], key_valid, last}
    reg [30:0] meta_s1_q, meta_s2_q;
    reg [95:0] shamt_s1_q, shamt_s2_q;
    reg [31:0] range_s1_q, range_s2_q;
    reg [15:0] tails_s1_q, tails_s2_q; // {low_code, high_code}
    reg [127:0] skip_s2_q;

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            valid_s1_q <= 1'b0;
            meta_s1_q  <= 31'd0;
            meta_s2_q  <= 31'd0;
            shamt_s1_q <= 96'd0;
            shamt_s2_q <= 96'd0;
            range_s1_q <= 32'd0;
            range_s2_q <= 32'd0;
            tails_s1_q <= 16'd0;
            tails_s2_q <= 16'd0;
            skip_s2_q  <= 128'd0;
        end else begin
            valid_s1_q <= valid_s0;
            if (valid_s0) begin
                meta_s1_q <= {feature_s0, token_tile_s0_q, lane_valid_s0,
                              key_valid_s0_q, last_s0_q};
                shamt_s1_q <= shamt_s0;
                range_s1_q <= range_s0;
                tails_s1_q <= {low_code, high_code};
            end
            if (valid_s1_q) begin
                meta_s2_q  <= meta_s1_q;
                shamt_s2_q <= shamt_s1_q;
                range_s2_q <= range_s1_q;
                tails_s2_q <= tails_s1_q;
                skip_s2_q  <= skip_s1_i;
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
        = meta_s2_q;

    // S1 input/control registers and S2 PREG are inside the DSP16 stage.
    vfu_dsp16 u_dsp (
        .clk_i   (clk_i),
        .rst_ni  (rst_ni),
        .ce_i    (1'b1),
        .valid_i (valid_s0),
        .op_i    (op_s0),
        .a_i     (a_s0),
        .b_i     (dsp_b),
        .c_i     (dsp_c),
        .valid_o (valid_s2),
        .op_o    (op_s2),
        .p_o     (p_s2)
    );

    // POST16 owns the S3 boundary, including final Moment S/Q capture.
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
        .shamt_i          (shamt_s2_q),
        .range_i          (range_s2_q),
        .low_code_i       (tails_s2_q[15:8]),
        .high_code_i      (tails_s2_q[7:0]),
        .skip_i           (skip_s2_q),
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

// Intentionally empty NN-LUT / coefficient-page wrapper.
// Header outputs depend on the selected phase page, not on op_s0_i.
// Lookup outputs are combinational from page + registered S0 op/feature/segment.
// RQ/RQ_RES/LN_AFFINE broadcast common-feature coefficients to all lanes;
// GELU/QEXP/RECIP/RSQRT select coefficients independently by lane segment.
// No output is driven yet: coefficient-dependent results are undefined.
module vfu_coeff_page16_draft #(
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
    // TODO: active-page storage and combinational coefficient lookup.
endmodule
