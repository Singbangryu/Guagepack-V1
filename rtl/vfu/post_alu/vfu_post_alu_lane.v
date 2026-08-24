`timescale 1ns/1ps
`include "vfu_defs.vh"

// =============================================================================
// GaugePack VFU S3 POST-ALU lane
// =============================================================================
// This is the source-of-truth single-lane POST datapath. A 16-lane wrapper
// supplies one independent shamt_i per lane.
//
// Quantized path:
//   P:S48 -> signed RNE right shift -> typed clamp -> output formatting
//
// Residual path:
//   P:S48 -> RNE -> NARROW_S8(main) -> exact main8+skip8 -> raw S9
//
// MOMENT and LN_NORM use the raw P bypass. PWL tail codes are already terminal
// output codes and therefore bypass the RNE/clamp middle-code datapath.
// =============================================================================

module vfu_post_alu_lane (
    input  wire [3:0]              op_i,
    input  wire signed [47:0]      p_i,
    input  wire [5:0]              shamt_i,

    // 00=MIDDLE, 01=LOW, 10=HIGH. 11 is treated as MIDDLE.
    input  wire [1:0]              range_i,

    // Compiler-produced terminal tail codes for GELU and LN_RSQRT.
    // QEXP tails are architecturally fixed to LOW=0 and HIGH=127.
    input  wire [7:0]              low_code_i,
    input  wire [7:0]              high_code_i,

    // Scalar validity of the current key.  The 16-lane wrapper broadcasts the
    // same bit to every query lane.  Used only by QEXP: invalid key -> E=0.
    input  wire                    key_valid_i,

    // Semantic zero override used only by LN_RSQRT (D=0).
    // Softmax reciprocal has no L=0 architectural state.
    input  wire                    force_zero_i,

    // Used only by RQ_RES; already in the same scale as quantized main8.
    input  wire signed [7:0]       skip_i,

    // Operation-normalized S32 container for scratch/writeback.
    output reg  [31:0]             data_o,

    // MomentPack field view. Capture/last-beat qualification belongs outside.
    output wire signed [15:0]      moment_s_o,
    output wire [22:0]             moment_q_o
);

    localparam [1:0] RANGE_LOW  = 2'b01;
    localparam [1:0] RANGE_HIGH = 2'b10;

    wire signed [47:0] rounded_w;
    wire        [1:0]  clamp_mode_w;
    wire        [7:0]  middle_code_w;
    wire signed [8:0]  residual_sum_w;
    wire signed [47:0] moment_s_full_w;

    wire [7:0] gelu_code_w;
    wire [7:0] qexp_code_w;
    wire [7:0] rsqrt_code_w;

    // Only QEXP and RSQRT require unsigned clamps. All other quantized
    // activation boundaries use the frozen symmetric S8 domain.
    assign clamp_mode_w =
        (op_i == `VFU_OP_QEXP)     ? `VFU_CLAMP_U7 :
        (op_i == `VFU_OP_LN_RSQRT) ? `VFU_CLAMP_U8 :
                                      `VFU_CLAMP_S8_SYM;

    vfu_rne_shift48 u_rne_shift (
        .x_i     (p_i),
        .shamt_i (shamt_i),
        .y_o     (rounded_w)
    );

    vfu_clamp_wrapper u_clamp (
        .x_i          (rounded_w),
        .clamp_mode_i (clamp_mode_w),
        .y_o          (middle_code_w)
    );

    vfu_residual_add u_residual_add (
        .main_i (middle_code_w),
        .skip_i (skip_i),
        .sum_o  (residual_sum_w)
    );

    // Tail codes are final output codes, not high-precision arithmetic data.
    assign gelu_code_w =
        (range_i == RANGE_LOW)  ? low_code_i  :
        (range_i == RANGE_HIGH) ? high_code_i :
                                  middle_code_w;

    // Frozen QEXP tails are exact architectural endpoints. If d==0 is encoded
    // as RANGE_MIDDLE, the compiled middle PWL result must be bit-exact 127.
    assign qexp_code_w =
        (range_i == RANGE_LOW)  ? 8'd0   :
        (range_i == RANGE_HIGH) ? 8'd127 :
                                  middle_code_w;

    assign rsqrt_code_w =
        (range_i == RANGE_LOW)  ? low_code_i  :
        (range_i == RANGE_HIGH) ? high_code_i :
                                  middle_code_w;

    // MomentPack stores P=(S<<23)+Q with Q in the low 23 bits.
    assign moment_s_full_w = $signed(p_i) >>> 23;
    assign moment_s_o      = moment_s_full_w[15:0];
    assign moment_q_o      = p_i[22:0];

    always @(*) begin
        data_o = 32'd0;

        case (op_i)
            `VFU_OP_RQ: begin
                data_o = {{24{middle_code_w[7]}}, middle_code_w};
            end

            `VFU_OP_RQ_RES: begin
                // Exact S9 result. Do not clamp after the residual add.
                data_o = {{23{residual_sum_w[8]}}, residual_sum_w};
            end

            `VFU_OP_GELU: begin
                data_o = {{24{gelu_code_w[7]}}, gelu_code_w};
            end

            `VFU_OP_QEXP: begin
                data_o = key_valid_i ? {24'd0, qexp_code_w} : 32'd0;
            end

            `VFU_OP_SM_RECIP_RAW: begin
                // Legal L is U13 [127,8128], so L=0 needs no override path.
                // Positive S18 by range certificate; preserve signed format.
                data_o = {{14{rounded_w[17]}}, rounded_w[17:0]};
            end

            `VFU_OP_SM_CONTEXT: begin
                data_o = {{24{middle_code_w[7]}}, middle_code_w};
            end

            `VFU_OP_LN_MOMENT_INIT,
            `VFU_OP_LN_MOMENT_ACC: begin
                // Architectural state is exposed through moment_s/q_o.
                data_o = 32'd0;
            end

            `VFU_OP_LN_D: begin
                // Nonnegative U27 by range certificate.
                data_o = {5'd0, rounded_w[26:0]};
            end

            `VFU_OP_LN_RSQRT: begin
                data_o = force_zero_i ? 32'd0 : {24'd0, rsqrt_code_w};
            end

            `VFU_OP_LN_NORM: begin
                // Raw T:S25; no shift, RNE, saturation, or requantization.
                data_o = {{7{p_i[24]}}, p_i[24:0]};
            end

            `VFU_OP_LN_AFFINE: begin
                data_o = {{24{middle_code_w[7]}}, middle_code_w};
            end

            default: begin
                data_o = 32'd0;
            end
        endcase
    end

endmodule
