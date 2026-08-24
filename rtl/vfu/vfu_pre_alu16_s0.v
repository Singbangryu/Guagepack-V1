`timescale 1ns/1ps
`include "vfu_defs.vh"

// =============================================================================
// GaugePack VFU PRE_ALU16 -- S0
// =============================================================================

module vfu_pre_alu_lane (
    input  wire [3:0]         op_i,
    input  wire signed [31:0] src0_i,
    input  wire signed [31:0] src1_i,
    input  wire        [7:0]  rsqrted_i,

    input  wire [404:0]       boundary_flat_i,
    input  wire signed [26:0] x_min_i,
    input  wire signed [26:0] x_max_i,

    output reg  signed [26:0] a_o,
    output reg  signed [17:0] b_o,
    output reg  signed [47:0] c_o,
    output wire        [3:0]  seg_addr_o,
    output wire        [1:0]  seg_range_o
);
    // Internal semantic op -> small PRE operand-pattern decode.
    localparam [2:0] PRE_PASS    = 3'd0; // A=src0
    localparam [2:0] PRE_PASS_B  = 3'd1; // A=src0, B=src1
    localparam [2:0] PRE_QEXP    = 3'd2; // A=src0-src1
    localparam [2:0] PRE_MOMENT  = 3'd3; // A=wire(2^23+z), B=z
    localparam [2:0] PRE_LN_D    = 3'd4; // A=S, B=-S, C=Q<<7; DSP performs C+A*B
    localparam [2:0] PRE_LN_NORM = 3'd5; // A=(z<<7)-S, B=rho


/*
                     A source                  B source                  C source                  DSP operation
-----------------------------------------------------------------------------------------------------------------------
RQ                   PRE.A = accumulator       M_RQ                      ZERO                      A*B + C = A*M_RQ
RQ_RES               PRE.A = accumulator       M_RQ                      ZERO                      A*B + C = A*M_RQ

GELU                 PRE.A = x                 NN_M[segment]             NN_C[segment]             A*B + C
QEXP                 PRE.A = score-rowmax      NN_M[segment]             NN_C[segment]             A*B + C
SM_RECIP_RAW         PRE.A = L                 NN_M[segment]             NN_C[segment]             A*B + C
LN_RSQRT             PRE.A = D                 NN_M[segment]             NN_C[segment]             A*B + C

SM_CONTEXT            PRE.A = N                PRE.B = R             ZERO                      A*B
LN_MOMENT_INIT        PRE.A = 2^23 + z         PRE.B = z             ZERO                      A*B
LN_MOMENT_ACC         PRE.A = 2^23 + z         PRE.B = z             P feedback                P + A*B

LN_D                  PRE.A = S                PRE.B = -S            PRE.C = Q<<7           C + A*B
LN_NORM               PRE.A = (z<<7)-S         PRE.B = rho           ZERO                      A*B

LN_AFFINE             PRE.A = T                M_gamma[feature]          C_beta[feature]            A*B + C
*/

    reg [2:0] pre_op;
    //pre-op decode VFU_OP is VFU's Global Op-code
    always @(*) begin
        case (op_i)
            `VFU_OP_RQ,
            `VFU_OP_RQ_RES,
            `VFU_OP_GELU,
            `VFU_OP_SM_RECIP_RAW,
            `VFU_OP_LN_RSQRT,
            `VFU_OP_LN_AFFINE:
                pre_op = PRE_PASS;

            `VFU_OP_SM_CONTEXT:
                pre_op = PRE_PASS_B;

            `VFU_OP_QEXP:
                pre_op = PRE_QEXP;

            `VFU_OP_LN_MOMENT_INIT,
            `VFU_OP_LN_MOMENT_ACC:
                pre_op = PRE_MOMENT;

            `VFU_OP_LN_D:
                pre_op = PRE_LN_D;

            `VFU_OP_LN_NORM:
                pre_op = PRE_LN_NORM;

            default:
                pre_op = PRE_PASS;
        endcase
    end

    // -------------------------------------------------------------------------
    // ONE shared signed subtractor.
    //
    // QEXP frozen semantic score width is S24, so score-rowmax needs S25.
    // LN_NORM needs only S17, and is sign-extended into the same S25 subtractor.
    // 1 subtractor per lane.. x 16 lane
    // -------------------------------------------------------------------------
    reg  signed [24:0] sub_a_w;
    reg  signed [24:0] sub_b_w;
    wire signed [24:0] sub_y_w = $signed(sub_a_w) - $signed(sub_b_w);

    always @(*) begin
        sub_a_w = 25'sd0;
        sub_b_w = 25'sd0;

        case (pre_op)
            PRE_QEXP: begin
                // src0/src1 carry sign-extended S24 score / rowmax.
                sub_a_w = {src0_i[23], src0_i[23:0]};
                sub_b_w = {src1_i[23], src1_i[23:0]};
            end

            PRE_LN_NORM: begin
                // 128*z
                // z:S9 -> (z<<7):S16 -> sign-extend to S25.
                sub_a_w = {{9{src0_i[8]}}, src0_i[8:0], 7'b0};

                // s (accum of z)
                sub_b_w = {{9{src1_i[15]}}, src1_i[15:0]};
            end

            default: begin
                sub_a_w = 25'sd0;
                sub_b_w = 25'sd0; // All other ops: shared subtractor inactive.
            end
        endcase
    end

    /*
     z >= 0 : 0x800000 + z
     z <  0 : 0x800000 - |z|
     for later... z^23 bit is neg_sign z_w... because if z_w is pos, z^32 is always 1,
        if z_w is neg, z^23 is always 0.
    z^23 is 0b1000_0000.....0000 so z_w's extend sign bit is propagted to z^23 positions.
        MSBS are always Zero...
    */
    wire signed [8:0] z_w = src0_i[8:0];
    wire sign_z;
    assign sign_z = z_w[8];
    wire signed [26:0] moment_a_w = {3'b000,~sign_z,{14{sign_z}},z_w};

    // Segment selector input.
    // For non-segmented ops this value is ignored by S1; no output gating mux
    // is added here.
    reg signed [31:0] seg_x_w;

// -------------------------------------------------------------------------
// Operand constructor
// -------------------------------------------------------------------------
    always @(*) begin
        a_o       = 27'sd0;
        b_o   = 18'sd0;
        c_o   = 48'sd0;
        seg_x_w   = 32'sd0;

        case (pre_op)
            PRE_PASS: begin
                // RQ/GELU/RECIP/RSQRT/AFFINE:
                a_o     = src0_i[26:0];
                seg_x_w = src0_i;
                //b_0,c_0 = 0
            end

            PRE_PASS_B: begin
                // Softmax context:
                //   A = N (S21 semantic, S32 container)
                //   B = R (positive S18)(signed-extend recip)
                //   C = 0
                a_o     = src0_i[26:0];
                b_o = src1_i[17:0];
                //c_0 = 0
            end

            PRE_QEXP: begin
                // d = score - rowmax, S25 semantic.
                // d stays in the raw QK integer domain.  The selected QEXP
                // page M/C already folds s_Q*s_K/sqrt(head_dim); do not add
                // a runtime >>3 for the BERT-Tiny head_dim=64 case.
                a_o     = {{2{sub_y_w[24]}}, sub_y_w};
                seg_x_w = {{7{sub_y_w[24]}}, sub_y_w};
            end

            PRE_MOMENT: begin
                // INIT and ACC have identical operands.
                // S2 distinguishes C-source vs P-feedback using op_i.
                a_o     = moment_a_w;
                b_o = {{9{z_w[8]}}, z_w};
            end

            PRE_LN_D: begin
                // FINAL:
                //   A = S
                //   B = -S
                //   C = Q<<7
                // S2 DSP48E2 performs C + A*B.
                a_o     = {{11{src0_i[15]}}, src0_i[15:0]};
                b_o     = -$signed({{2{src0_i[15]}}, src0_i[15:0]});
                c_o = {18'd0, src1_i[22:0], 7'b0};
            end

            PRE_LN_NORM: begin
                // n = (z<<7)-S, produced by the shared subtractor.
                // rho is the only third operand in PRE, so it uses a narrow U8 bus.
                a_o     = {{2{sub_y_w[24]}}, sub_y_w};
                b_o = {10'd0, rsqrted_i};
            end

            default: begin
                a_o       = 27'sd0;
                b_o   = 18'sd0;
                c_o   = 48'sd0;
                seg_x_w   = 32'sd0;
            end
        endcase
    end

    gaugepack_vfu_segment_gen u_segment_gen (
        .x_i             (seg_x_w),
        .boundary_flat_i (boundary_flat_i),
        .x_min_i         (x_min_i),
        .x_max_i         (x_max_i),
        .addr_o          (seg_addr_o),
        .range_o         (seg_range_o)
    );

endmodule


module vfu_pre_alu16 (
    input  wire                    clk_i,
    input  wire                    rst_ni,
    input  wire                    ce_i,

    input  wire                    valid_i,
    input  wire [3:0]              op_i,
    input  wire [8:0]              feature_i,
    input  wire [15:0]             lane_valid_i,

    input  wire [511:0]            src0_i,       // 16 x S32
    input  wire [511:0]            src1_i,       // 16 x S32
    input  wire [127:0]            rsqrted_i,    // 16 x U8; LN rho

    // Segment-page metadata is supplied from the coefficient/page side.
    // page_id itself is NOT pipelined through PRE; top/FSM owns it.
    input  wire [404:0]            boundary_flat_i, // 15 x S27
    input  wire signed [26:0]      x_min_i,
    input  wire signed [26:0]      x_max_i,

    output reg                     valid_o,
    output reg  [3:0]              op_o,
    output reg  [8:0]              feature_o,
    output reg  [15:0]             lane_valid_o,

    output reg  [431:0]            a_o,          // 16 x S27
    output reg  [287:0]            b_o,          // 16 x S18
    output reg  [767:0]            c_o,          // 16 x S48
    output reg  [63:0]             seg_addr_o,   // 16 x 4
    output reg  [31:0]             seg_range_o   // 16 x 2
);

    wire [431:0] a_w;
    wire [287:0] b_w;
    wire [767:0] c_w;
    wire [63:0]  seg_addr_w;
    wire [31:0]  seg_range_w;

    genvar lane;
    generate
        for (lane = 0; lane < 16; lane = lane + 1) begin : GEN_PRE_LANE
            vfu_pre_alu_lane u_pre_lane (
                .op_i            (op_i),
                .src0_i          (src0_i[lane*32 +: 32]),
                .src1_i          (src1_i[lane*32 +: 32]),
                .rsqrted_i       (rsqrted_i[lane*8 +: 8]),
                .boundary_flat_i (boundary_flat_i),
                .x_min_i         (x_min_i),
                .x_max_i         (x_max_i),
                .a_o             (a_w[lane*27 +: 27]),
                .b_o             (b_w[lane*18 +: 18]),
                .c_o             (c_w[lane*48 +: 48]),
                .seg_addr_o      (seg_addr_w[lane*4 +: 4]),
                .seg_range_o     (seg_range_w[lane*2 +: 2])
            );
        end
    endgenerate

    // Registered S0 PRE stage.
    // Keep only sideband that must stay beat-aligned with the PRE operands.
    // page_id / layer / site / quant-profile state is held by top/FSM.
    always @(posedge clk_i) begin
        if (!rst_ni) begin
            valid_o      <= 1'b0;
            op_o         <= 4'd0;
            feature_o    <= 9'd0;
            lane_valid_o <= 16'd0;
            a_o          <= 432'd0;
            b_o          <= 288'd0;
            c_o          <= 768'd0;
            seg_addr_o   <= 64'd0;
            seg_range_o  <= 32'd0;
        end else if (ce_i) begin
            valid_o      <= valid_i;
            op_o         <= op_i;
            feature_o    <= feature_i;
            lane_valid_o <= lane_valid_i;

            if (valid_i) begin
                a_o         <= a_w;
                b_o         <= b_w;
                c_o         <= c_w;
                seg_addr_o  <= seg_addr_w;
                seg_range_o <= seg_range_w;
            end
        end
    end

endmodule
