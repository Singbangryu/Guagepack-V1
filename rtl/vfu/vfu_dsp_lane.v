`timescale 1ns/1ps
`include "vfu_defs.vh"

// =============================================================================
// GaugePack VFU DSP lane
// =============================================================================
// Physical pipeline placement:
//
//   S0 PRE register
//        |
//        |  S1 combinational coefficient / operand select
//        v
//   DSP48E2 AREG/BREG/CREG + OPMODEREG/ALUMODEREG   <-- S1 register
//        |
//        |  MREG=0 : multiplier + 48-bit ALU
//        v
//   DSP48E2 PREG                                      <-- S2 register
//        |
//        v
//   S3 POST
//
// Frozen arithmetic modes:
//   normal      : P = C + A*B
//   LN_D        : P = C + A*B, with PRE supplying B=-S
//   MOMENT_ACC  : P = P + A*B
//
// MOMENT_INIT uses the normal path with C=0.
// SM_CONTEXT / LN_NORM use the normal path with C=0.
//
// Important:
//   - PRE already constructs the correct A/B/C.
//   - coefficient selection belongs to S1, not here.
//   - no RNE / shift / saturation belongs here.
// =============================================================================

module vfu_dsp_lane (
    input  wire                    clk_i,
    input  wire                    rst_ni,
    input  wire                    ce_i,

    // Current S0/S1 item.  These are captured into the DSP's own input/control
    // registers, which physically implement the S1 register boundary.
    input  wire                    valid_i,
    input  wire [3:0]              op_i,
    input  wire signed [26:0]      a_i,
    input  wire signed [17:0]      b_i,
    input  wire signed [47:0]      c_i,

    // PREG output: architectural S2 result.
    output reg                     valid_o,
    output reg  [3:0]              op_o,
    output wire signed [47:0]      p_o
);

    // -------------------------------------------------------------------------
    // DSP48E2 instruction decode.
    //
    // OPMODE fields:
    //   [8:7] W = 00 : 0
    //   [6:4] Z = 011: C, 010: P
    //   [3:2] Y = 01 : multiplier partial product M
    //   [1:0] X = 01 : multiplier partial product M
    //
    // Thus:
    //   00_011_01_01 = 9'b000110101 : Z=C, XY=M
    //   00_010_01_01 = 9'b000100101 : Z=P, XY=M
    //
    // ALUMODE:
    //   0000 : Z + (W+X+Y+CIN)
    //   0011 : Z - (W+X+Y+CIN)
    // -------------------------------------------------------------------------
    localparam [8:0] OPMODE_C_PLUS_M = 9'b000110101;
    localparam [8:0] OPMODE_P_PLUS_M = 9'b000100101;

    localparam [3:0] ALUMODE_ADD      = 4'b0000;

    reg [8:0] opmode_w;
    reg [3:0] alumode_w;

    always @(*) begin
        // Default: C + A*B
        opmode_w  = OPMODE_C_PLUS_M;
        alumode_w = ALUMODE_ADD;

        case (op_i)
            `VFU_OP_LN_MOMENT_ACC: begin
                // Stateful MomentPack accumulation:
                // P <- P + A*B
                opmode_w  = OPMODE_P_PLUS_M;
                alumode_w = ALUMODE_ADD;
            end

            `VFU_OP_LN_D: begin
                // D = (Q<<7) - S*S.
                // Frozen operand contract: PRE supplies A=S, B=-S, C=Q<<7.
                opmode_w  = OPMODE_C_PLUS_M;
                alumode_w = ALUMODE_ADD;
            end

            default: begin
                // RQ / RQ_RES / GELU / QEXP / RECIP / CONTEXT /
                // MOMENT_INIT / RSQRT / LN_NORM / LN_AFFINE
                opmode_w  = OPMODE_C_PLUS_M;
                alumode_w = ALUMODE_ADD;
            end
        endcase
    end

    // DSP A port is 30 bits; the multiplier consumes signed A27.
    wire [29:0] dsp_a_w = {{3{a_i[26]}}, a_i};
    wire [17:0] dsp_b_w = b_i;
    wire [47:0] dsp_c_w = c_i;

    wire rst_w = ~rst_ni;

    // Capture only real items into S1.  A bubble leaves the DSP input/control
    // registers untouched and is represented only by valid_s1_q=0.
    wire ce_s1_w = ce_i && valid_i;

    // PREG must update only when the item currently resident in S1 is valid.
    // This is important for MOMENT_ACC: a bubble must hold P, not accumulate
    // stale operands again.
    reg valid_s1_q;
    reg [3:0] op_s1_q;

    wire ce_p_w = ce_i && valid_s1_q;

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            valid_s1_q <= 1'b0;
            op_s1_q    <= 4'd0;
            valid_o    <= 1'b0;
            op_o       <= 4'd0;
        end else if (ce_i) begin
            valid_s1_q <= valid_i;
            valid_o    <= valid_s1_q;

            if (valid_i)
                op_s1_q <= op_i;

            if (valid_s1_q)
                op_o <= op_s1_q;
        end
    end

    // -------------------------------------------------------------------------
    // DSP48E2 primitive
    // -------------------------------------------------------------------------
    // AREG/BREG/CREG/OPMODEREG/ALUMODEREG = 1 implement S1.
    // MREG = 0 keeps MUL+ALU in one combinational DSP stage.
    // PREG = 1 implements S2.
    //
    // INMODE=00000 selects A2/B2 with AREG=BREG=1 and bypasses the pre-adder.
    // -------------------------------------------------------------------------
    wire [47:0] dsp_p_w;

    DSP48E2 #(
        // Datapath selection
        .AMULTSEL("A"),
        .A_INPUT("DIRECT"),
        .BMULTSEL("B"),
        .B_INPUT("DIRECT"),
        .PREADDINSEL("A"),
        .RND(48'h000000000000),
        .USE_MULT("MULTIPLY"),
        .USE_SIMD("ONE48"),
        .USE_WIDEXOR("FALSE"),
        .XORSIMD("XOR24_48_96"),

        // Input/control/output pipeline
        .ACASCREG(1),
        .ADREG(0),
        .ALUMODEREG(1),
        .AREG(1),

        .BCASCREG(1),
        .BREG(1),

        .CARRYINREG(0),
        .CARRYINSELREG(1),
        .CREG(1),

        .DREG(0),
        .INMODEREG(0),
        .MREG(0),
        .OPMODEREG(1),
        .PREG(1),

        // Pattern detector unused
        .AUTORESET_PATDET("NO_RESET"),
        .AUTORESET_PRIORITY("RESET"),
        .MASK(48'h3fffffffffff),
        .PATTERN(48'h000000000000),
        .SEL_MASK("MASK"),
        .SEL_PATTERN("PATTERN"),
        .USE_PATTERN_DETECT("NO_PATDET"),

        // No programmable input inversion
        .IS_ALUMODE_INVERTED(4'b0000),
        .IS_CARRYIN_INVERTED(1'b0),
        .IS_CLK_INVERTED(1'b0),
        .IS_INMODE_INVERTED(5'b00000),
        .IS_OPMODE_INVERTED(9'b000000000),
        .IS_RSTALLCARRYIN_INVERTED(1'b0),
        .IS_RSTALUMODE_INVERTED(1'b0),
        .IS_RSTA_INVERTED(1'b0),
        .IS_RSTB_INVERTED(1'b0),
        .IS_RSTCTRL_INVERTED(1'b0),
        .IS_RSTC_INVERTED(1'b0),
        .IS_RSTD_INVERTED(1'b0),
        .IS_RSTINMODE_INVERTED(1'b0),
        .IS_RSTM_INVERTED(1'b0),
        .IS_RSTP_INVERTED(1'b0)
    ) u_dsp48e2 (
        // Cascade / status outputs unused
        .ACOUT(),
        .BCOUT(),
        .CARRYCASCOUT(),
        .CARRYOUT(),
        .MULTSIGNOUT(),
        .OVERFLOW(),
        .PATTERNBDETECT(),
        .PATTERNDETECT(),
        .PCOUT(),
        .UNDERFLOW(),
        .XOROUT(),

        // Architectural S2 result
        .P(dsp_p_w),

        // Cascade inputs unused
        .ACIN(30'd0),
        .BCIN(18'd0),
        .CARRYCASCIN(1'b0),
        .MULTSIGNIN(1'b0),
        .PCIN(48'd0),

        // Dynamic operation controls
        .ALUMODE(alumode_w),
        .CARRYIN(1'b0),
        .CARRYINSEL(3'b000),
        .INMODE(5'b00000),
        .OPMODE(opmode_w),

        // Data
        .A(dsp_a_w),
        .B(dsp_b_w),
        .C(dsp_c_w),
        .D(27'd0),

        .CLK(clk_i),

        // S1 CEs: DSP input + control registers
        .CEA1(1'b0),
        .CEA2(ce_s1_w),
        .CEAD(1'b0),
        .CEALUMODE(ce_s1_w),

        .CEB1(1'b0),
        .CEB2(ce_s1_w),

        .CEC(ce_s1_w),
        .CECARRYIN(1'b0),
        .CECTRL(ce_s1_w),
        .CED(1'b0),
        .CEINMODE(1'b0),
        .CEM(1'b0),

        // S2 P register
        .CEP(ce_p_w),

        // Synchronous resets
        .RSTA(rst_w),
        .RSTALLCARRYIN(rst_w),
        .RSTALUMODE(rst_w),
        .RSTB(rst_w),
        .RSTC(rst_w),
        .RSTCTRL(rst_w),
        .RSTD(rst_w),
        .RSTINMODE(rst_w),
        .RSTM(rst_w),
        .RSTP(rst_w)
    );

    assign p_o = dsp_p_w;

endmodule
