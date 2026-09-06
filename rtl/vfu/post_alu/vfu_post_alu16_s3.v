`timescale 1ns/1ps
`include "vfu_internal_op_defs.vh"

// Sixteen inlined POST lanes followed by the architectural S3 registers.
// One shamt/range per lane; tail codes and key_valid are common to the beat.
// All outputs are S3-aligned. No address, packing-to-ACT16, or persistent
// per-tile state bank is owned here. Lane occupancy never gates beat valid.
module vfu_post_alu16_s3 (
    input  wire         clk_i,
    input  wire         rst_ni,
    input  wire         valid_i,
    input  wire [3:0]   op_i,
    input  wire [8:0]   feature_i,
    input  wire [3:0]   token_tile_i,
    input  wire [15:0]  lane_valid_i,
    input  wire         key_valid_i,
    input  wire         last_i,
    input  wire [767:0] p_i,
    input  wire [95:0]  shamt_i,
    input  wire [31:0]  range_i,
    input  wire [7:0]   low_code_i,
    input  wire [7:0]   high_code_i,
    input  wire [127:0] skip_i,

    output reg          valid_o,
    output reg  [3:0]   op_o,
    output reg  [8:0]   feature_o,
    output reg  [3:0]   token_tile_o,
    output reg  [15:0]  lane_valid_o,
    output reg          key_valid_o,
    output reg          last_o,
    output reg  [511:0] data_o,       // 16 x operation-specific S32/U32
    output reg          moment_capture_o,
    output reg  [255:0] moment_s_o,   // 16 x S16
    output reg  [367:0] moment_q_o    // 16 x U23
);
    wire [511:0] post_data;
    wire [255:0] post_moment_s;
    wire [367:0] post_moment_q;

    genvar lane;
    generate
        for (lane = 0; lane < 16; lane = lane + 1) begin : GEN_LANE
            wire signed [47:0] p_lane = p_i[lane*48 +: 48];
            wire [5:0] shamt_lane = shamt_i[lane*6 +: 6];
            wire [1:0] range_lane = range_i[lane*2 +: 2];
            wire signed [7:0] skip_lane = skip_i[lane*8 +: 8];
            reg [31:0] data_lane;
            wire signed [15:0] moment_s_lane;
            wire [22:0] moment_q_lane;
            // Current LN contract has no D==0 override.
            wire force_zero_i = 1'b0;

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
                .x_i     (p_lane),
                .shamt_i (shamt_lane),
                .y_o     (rounded_w)
            );

            vfu_clamp_wrapper u_clamp (
                .x_i          (rounded_w),
                .clamp_mode_i (clamp_mode_w),
                .y_o          (middle_code_w)
            );

            vfu_residual_add u_residual_add (
                .main_i (middle_code_w),
                .skip_i (skip_lane),
                .sum_o  (residual_sum_w)
            );

            // Tail codes are final output codes, not high-precision arithmetic data.
            assign gelu_code_w =
                (range_lane == RANGE_LOW)  ? low_code_i  :
                (range_lane == RANGE_HIGH) ? high_code_i :
                                          middle_code_w;

            // Frozen QEXP tails are exact architectural endpoints. If d==0 is encoded
            // as RANGE_MIDDLE, the compiled middle PWL result must be bit-exact 127.
            assign qexp_code_w =
                (range_lane == RANGE_LOW)  ? 8'd0   :
                (range_lane == RANGE_HIGH) ? 8'd127 :
                                          middle_code_w;

            assign rsqrt_code_w =
                (range_lane == RANGE_LOW)  ? low_code_i  :
                (range_lane == RANGE_HIGH) ? high_code_i :
                                          middle_code_w;

            // MomentPack stores P=(S<<23)+Q with Q in the low 23 bits.
            assign moment_s_full_w = $signed(p_lane) >>> 23;
            assign moment_s_lane      = moment_s_full_w[15:0];
            assign moment_q_lane      = p_lane[22:0];

            always @(*) begin
                data_lane = 32'd0;

                case (op_i)
                    `VFU_OP_RQ: begin
                        data_lane = {{24{middle_code_w[7]}}, middle_code_w};
                    end

                    `VFU_OP_RQ_RES: begin
                        // Exact S9 result. Do not clamp after the residual add.
                        data_lane = {{23{residual_sum_w[8]}}, residual_sum_w};
                    end

                    `VFU_OP_GELU: begin
                        data_lane = {{24{gelu_code_w[7]}}, gelu_code_w};
                    end

                    `VFU_OP_QEXP: begin
                        data_lane = key_valid_i ? {24'd0, qexp_code_w} : 32'd0;
                    end

                    `VFU_OP_SM_RECIP_RAW: begin
                        // Legal L is U13 [127,8128], so L=0 needs no override path.
                        // Positive S18 by range certificate; preserve signed format.
                        data_lane = {{14{rounded_w[17]}}, rounded_w[17:0]};
                    end

                    `VFU_OP_SM_CONTEXT: begin
                        data_lane = {{24{middle_code_w[7]}}, middle_code_w};
                    end

                    `VFU_OP_LN_MOMENT_INIT,
                    `VFU_OP_LN_MOMENT_ACC: begin
                        // Architectural state is exposed through moment_s/q_o.
                        data_lane = 32'd0;
                    end

                    `VFU_OP_LN_D: begin
                        // Nonnegative U27 by range certificate.
                        data_lane = {5'd0, rounded_w[26:0]};
                    end

                    `VFU_OP_LN_RSQRT: begin
                        data_lane = force_zero_i ? 32'd0 : {24'd0, rsqrt_code_w};
                    end

                    `VFU_OP_LN_NORM: begin
                        // Raw T:S25; no shift, RNE, saturation, or requantization.
                        data_lane = {{7{p_lane[24]}}, p_lane[24:0]};
                    end

                    `VFU_OP_LN_AFFINE: begin
                        data_lane = {{24{middle_code_w[7]}}, middle_code_w};
                    end

                    default: begin
                        data_lane = 32'd0;
                    end
                endcase
            end

            assign post_data[lane*32 +: 32] = data_lane;
            assign post_moment_s[lane*16 +: 16] = moment_s_lane;
            assign post_moment_q[lane*23 +: 23] = moment_q_lane;
        end
    endgenerate

    wire capture_moment = valid_i && last_i &&
        ((op_i == `VFU_OP_LN_MOMENT_INIT) ||
         (op_i == `VFU_OP_LN_MOMENT_ACC));

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            valid_o          <= 1'b0;
            op_o             <= 4'd0;
            feature_o        <= 9'd0;
            token_tile_o     <= 4'd0;
            lane_valid_o     <= 16'd0;
            key_valid_o      <= 1'b0;
            last_o           <= 1'b0;
            data_o           <= 512'd0;
            moment_capture_o <= 1'b0;
            moment_s_o       <= 256'd0;
            moment_q_o       <= 368'd0;
        end else begin
            valid_o          <= valid_i;
            moment_capture_o <= capture_moment;
            if (valid_i) begin
                op_o         <= op_i;
                feature_o    <= feature_i;
                token_tile_o <= token_tile_i;
                lane_valid_o <= lane_valid_i;
                key_valid_o  <= key_valid_i;
                last_o       <= last_i;
                data_o       <= post_data;
            end
            if (capture_moment) begin
                moment_s_o <= post_moment_s;
                moment_q_o <= post_moment_q;
            end
        end
    end
endmodule
