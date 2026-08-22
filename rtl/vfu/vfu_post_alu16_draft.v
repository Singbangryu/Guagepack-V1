`timescale 1ns/1ps
`include "vfu_defs.vh"

// =============================================================================
// GaugePack VFU POST_ALU lane -- draft
// =============================================================================
// Input is the DSP48E2 PREG result (architectural S2).
//
// One physical shift/RNE path exists per lane.
// shift_i is common across all 16 lanes for one beat, but each lane owns its
// own data shifter/RNE logic.
//
// Final policy:
//   - RNE only at precision-reducing quantization boundaries.
//   - MOMENT and LN_NORM bypass shift/RNE.
//   - RQ_RES rounds/narrows main first, then adds original skip8 into raw S9.
//   - Tail selection happens after middle-code RNE/clamp.
//   - L==0 / D==0 detection is NOT implemented here; the aligned
//     force_zero_i sideband performs the final zero override.
//   - No terminal-GELU-LUT dual mode is implemented in this draft.
// =============================================================================

module vfu_post_alu_lane (
    input  wire [3:0]              op_i,
    input  wire signed [47:0]      p_i,
    input  wire [5:0]              shift_i,

    // Existing segment selector encoding:
    //   00 = MIDDLE, 01 = LOW, 10 = HIGH
    input  wire [1:0]              range_i,
    input  wire [7:0]              low_code_i,
    input  wire [7:0]              high_code_i,

    // QEXP invalid/masked pair -> E=0.
    input  wire                    pair_valid_i,

    // Aligned semantic override:
    //   SM_RECIP_RAW: L==0
    //   LN_RSQRT    : D==0
    input  wire                    force_zero_i,

    // Used only by RQ_RES.
    input  wire signed [7:0]       skip_i,

    // Op-normalized S32 result for the common write/loopback path.
    output reg  [31:0]             data_o,

    // MomentPack final-field view.  The wrapper qualifies these with
    // moment_capture_o on the final beat.
    output wire signed [15:0]      moment_s_o,
    output wire [22:0]             moment_q_o
);

    localparam [1:0] RANGE_MIDDLE = 2'b00;
    localparam [1:0] RANGE_LOW    = 2'b01;
    localparam [1:0] RANGE_HIGH   = 2'b10;

    // -------------------------------------------------------------------------
    // Signed round-to-nearest, ties-to-even:
    //
    //   q_floor = floor(value / 2^shift)
    //   rem     = value - q_floor*2^shift, therefore rem is nonnegative
    //   inc     = rem > half || (rem == half && q_floor is odd)
    //   result  = q_floor + inc
    //
    // For a two's-complement arithmetic right shift, rem is exactly the lower
    // shift bits.  Legal shift contract is 0..47.
    // -------------------------------------------------------------------------
    function signed [47:0] rne_shift_signed;
        input signed [47:0] value;
        input        [5:0]  shamt;

        reg signed [47:0] q_floor;
        reg        [47:0] mask;
        reg        [47:0] rem;
        reg        [47:0] half;
        reg               inc;
        begin
            if (shamt == 0) begin
                rne_shift_signed = value;
            end else begin
                q_floor = $signed(value) >>> shamt;
                mask    = (48'd1 << shamt) - 48'd1;
                rem     = value & mask;
                half    = 48'd1 << (shamt - 1'b1);

                inc = (rem > half)
                   || ((rem == half) && q_floor[0]);

                rne_shift_signed =
                    q_floor + {{47{1'b0}}, inc};
            end
        end
    endfunction

    // Symmetric narrow-S8: -128 is forbidden.
    function [7:0] sat_narrow_s8;
        input signed [47:0] value;
        begin
            if (value > 48'sd127)
                sat_narrow_s8 = 8'h7f;
            else if (value < -48'sd127)
                sat_narrow_s8 = 8'h81; // -127
            else
                sat_narrow_s8 = value[7:0];
        end
    endfunction

    function [7:0] clamp_e7;
        input signed [47:0] value;
        begin
            if (value < 0)
                clamp_e7 = 8'd0;
            else if (value > 48'sd127)
                clamp_e7 = 8'd127;
            else
                clamp_e7 = value[7:0];
        end
    endfunction

    function [7:0] clamp_u8;
        input signed [47:0] value;
        begin
            if (value < 0)
                clamp_u8 = 8'd0;
            else if (value > 48'sd255)
                clamp_u8 = 8'd255;
            else
                clamp_u8 = value[7:0];
        end
    endfunction

    function [7:0] clamp_tail_e7;
        input [7:0] value;
        begin
            clamp_tail_e7 = (value > 8'd127) ? 8'd127 : value;
        end
    endfunction

    wire signed [47:0] rounded_w =
        rne_shift_signed(p_i, shift_i);

    wire signed [47:0] moment_s_full_w =
        $signed(p_i) >>> 23;

    assign moment_s_o = moment_s_full_w[15:0];
    assign moment_q_o = p_i[22:0];

    reg  [7:0] middle_s8_w;
    reg  [7:0] middle_e7_w;
    reg  [7:0] middle_u8_w;
    reg  [7:0] pwl_byte_w;
    reg  [7:0] main8_w;
    reg signed [8:0] residual9_w;

    always @(*) begin
        middle_s8_w = sat_narrow_s8(rounded_w);
        middle_e7_w = clamp_e7(rounded_w);
        middle_u8_w = clamp_u8(rounded_w);

        pwl_byte_w  = 8'd0;
        main8_w     = 8'd0;
        residual9_w = 9'sd0;
        data_o      = 32'd0;

        case (op_i)
            `VFU_OP_RQ: begin
                // RNE(P/2^F_profile) -> NARROW_S8
                data_o = {{24{middle_s8_w[7]}}, middle_s8_w};
            end

            `VFU_OP_RQ_RES: begin
                // First quantize main into the native skip scale.
                main8_w = middle_s8_w;

                // Then exact common-scale residual add.  No post-add saturation.
                residual9_w =
                    $signed({main8_w[7], main8_w})
                  + $signed({skip_i[7], skip_i});

                data_o = {{23{residual9_w[8]}}, residual9_w};
            end

            `VFU_OP_GELU: begin
                // Middle segment: RNE + NARROW_S8.
                // Low/high tail bytes are compiler-produced signed-S8 codes.
                case (range_i)
                    RANGE_LOW:  pwl_byte_w = low_code_i;
                    RANGE_HIGH: pwl_byte_w = high_code_i;
                    default:    pwl_byte_w = middle_s8_w;
                endcase

                data_o = {{24{pwl_byte_w[7]}}, pwl_byte_w};
            end

            `VFU_OP_QEXP: begin
                // Middle segment: RNE + E7 clamp.
                // Tail bytes are also clamped to the architectural E7 domain.
                case (range_i)
                    RANGE_LOW:  pwl_byte_w = clamp_tail_e7(low_code_i);
                    RANGE_HIGH: pwl_byte_w = clamp_tail_e7(high_code_i);
                    default:    pwl_byte_w = middle_e7_w;
                endcase

                // Invalid/masked key never contributes to E scratch or rowsum.
                if (!pair_valid_i)
                    pwl_byte_w = 8'd0;

                data_o = {24'd0, pwl_byte_w};
            end

            `VFU_OP_SM_RECIP_RAW: begin
                // RNE(P/2^8) -> positive S18 by compiler/range certificate.
                // POST does not add a runtime S18 saturator in this draft.
                if (force_zero_i)
                    data_o = 32'd0;
                else
                    data_o = {{14{rounded_w[17]}}, rounded_w[17:0]};
            end

            `VFU_OP_SM_CONTEXT: begin
                // RNE(P/2^23) -> NARROW_S8
                data_o = {{24{middle_s8_w[7]}}, middle_s8_w};
            end

            `VFU_OP_LN_MOMENT_INIT,
            `VFU_OP_LN_MOMENT_ACC: begin
                // No shift or rounding.  Architectural state is captured through
                // moment_s_o/moment_q_o only on the wrapper's final beat.
                data_o = 32'd0;
            end

            `VFU_OP_LN_D: begin
                // RNE(P/2^4) -> nonnegative D27 by arithmetic proof.
                // No runtime D27 saturation in this draft.
                data_o = {5'd0, rounded_w[26:0]};
            end

            `VFU_OP_LN_RSQRT: begin
                // Middle segment: RNE + U8 clamp.
                case (range_i)
                    RANGE_LOW:  pwl_byte_w = low_code_i;
                    RANGE_HIGH: pwl_byte_w = high_code_i;
                    default:    pwl_byte_w = middle_u8_w;
                endcase

                if (force_zero_i)
                    pwl_byte_w = 8'd0;

                data_o = {24'd0, pwl_byte_w};
            end

            `VFU_OP_LN_NORM: begin
                // Raw T:S25.  No shift, RNE, saturation, or requant.
                data_o = {{7{p_i[24]}}, p_i[24:0]};
            end

            `VFU_OP_LN_AFFINE: begin
                // Final LayerNorm boundary: RNE(P/2^F_final) -> NARROW_S8
                data_o = {{24{middle_s8_w[7]}}, middle_s8_w};
            end

            default: begin
                data_o = 32'd0;
            end
        endcase
    end

endmodule


// =============================================================================
// 16-lane S3 wrapper
// =============================================================================
module vfu_post_alu16 (
    input  wire                    clk_i,
    input  wire                    rst_ni,
    input  wire                    ce_i,

    // Already aligned to the DSP PREG/S2 result.
    input  wire                    valid_i,
    input  wire [3:0]              op_i,
    input  wire [8:0]              feature_i,
    input  wire [15:0]             lane_valid_i,
    input  wire                    last_i,

    input  wire [767:0]            p_i,          // 16 x S48

    // One common shift amount for all 16 lanes in this beat.
    input  wire [5:0]              shift_i,

    // PWL metadata, aligned with P.
    input  wire [31:0]             range_i,      // 16 x 2
    input  wire [7:0]              low_code_i,   // common page tail
    input  wire [7:0]              high_code_i,  // common page tail

    // Semantic sidebands.
    input  wire [15:0]             pair_valid_i,
    input  wire [15:0]             force_zero_i,
    input  wire [127:0]            skip_i,       // 16 x S8; RQ_RES only

    // Registered S3 output.
    output reg                     valid_o,
    output reg  [3:0]              op_o,
    output reg  [8:0]              feature_o,
    output reg  [15:0]             lane_valid_o,
    output reg                     last_o,
    output reg  [511:0]            data_o,       // 16 x normalized S32

    // MomentPack architectural commit.
    output reg                     moment_capture_o,
    output reg  [255:0]            moment_s_o,   // 16 x S16
    output reg  [367:0]            moment_q_o    // 16 x U23
);

    wire [511:0] data_w;
    wire [255:0] moment_s_w;
    wire [367:0] moment_q_w;

    genvar lane;
    generate
        for (lane = 0; lane < 16; lane = lane + 1) begin : GEN_POST_LANE
            vfu_post_alu_lane u_post_lane (
                .op_i          (op_i),
                .p_i           (p_i[lane*48 +: 48]),
                .shift_i       (shift_i),
                .range_i       (range_i[lane*2 +: 2]),
                .low_code_i    (low_code_i),
                .high_code_i   (high_code_i),
                .pair_valid_i  (pair_valid_i[lane]),
                .force_zero_i  (force_zero_i[lane]),
                .skip_i        (skip_i[lane*8 +: 8]),
                .data_o        (data_w[lane*32 +: 32]),
                .moment_s_o    (moment_s_w[lane*16 +: 16]),
                .moment_q_o    (moment_q_w[lane*23 +: 23])
            );
        end
    endgenerate

    wire moment_op_w =
           (op_i == `VFU_OP_LN_MOMENT_INIT)
        || (op_i == `VFU_OP_LN_MOMENT_ACC);

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            valid_o          <= 1'b0;
            op_o             <= 4'd0;
            feature_o        <= 9'd0;
            lane_valid_o     <= 16'd0;
            last_o           <= 1'b0;
            data_o           <= 512'd0;
            moment_capture_o <= 1'b0;
            moment_s_o       <= 256'd0;
            moment_q_o       <= 368'd0;
        end else if (ce_i) begin
            valid_o          <= valid_i;
            moment_capture_o <= valid_i && last_i && moment_op_w;

            if (valid_i) begin
                op_o         <= op_i;
                feature_o    <= feature_i;
                lane_valid_o <= lane_valid_i;
                last_o       <= last_i;
                data_o       <= data_w;

                if (last_i && moment_op_w) begin
                    moment_s_o <= moment_s_w;
                    moment_q_o <= moment_q_w;
                end
            end
        end
    end

`ifndef SYNTHESIS
    // Contract check only; production controller/compiler must keep shift in 0..47.
    always @(posedge clk_i) begin
        if (rst_ni && ce_i && valid_i && (shift_i > 6'd47)) begin
            $display("ERROR: vfu_post_alu16 shift_i=%0d exceeds legal 0..47", shift_i);
            $stop;
        end
    end
`endif

endmodule
