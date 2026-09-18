`timescale 1ns/1ps
`include "vfu_internal_op_defs.vh"

// DRAFT: event scheduler, with no arithmetic or memory instantiated.
// Accept all MT*128 paired main/skip inputs before processing tiles serially.
// External adapters align operands and result tags, own scratch writes and
// qualify every completion for this command, tile and operation. All padded
// lanes progress. The final writer alone masks architectural padded bytes.
//
// Four always-accepted replay requests per tile: INIT(0,1), ACC(1,127),
// NORM(0,128), AFFINE(0,128). Returns arrive in ascending feature order,
// after their request, with no leftover response crossing a drain boundary.
// scalar_ready is an operand-adapter condition, NOT a new CORE ready port.
// Op/page must remain stable through each drain. INIT retirement is ordinary
// tagged S3 retirement; only final ACC last causes CORE Moment S/Q capture.
module vfu_ln_control_draft (
    input  wire       clk_i,
    input  wire       rst_ni,
    input  wire       init_i,
    input  wire       active_i,
    input  wire [2:0] mt_i,                 // Held count, 1..4; N is fixed 128.
    input  wire       rq_valid_i,           // Main/skip aligned for real launch.
    input  wire       z_store_done_i,
    input  wire       replay_valid_i,
    input  wire       init_retire_i,
    input  wire       moment_capture_i,
    input  wire       scalar_ready_i,
    input  wire       d_result_valid_i,
    input  wire       rho_store_done_i,
    input  wire       rho_match_valid_i,
    input  wire       t_store_done_i,
    input  wire       affine_tile_done_i,
    input  wire       commit_done_i,

    output wire       rq_en_o,
    output wire       rq_fire_o,
    output wire       rq_last_o,
    output wire       replay_start_o,
    output wire       replay_en_o,
    output wire       replay_fire_o,
    output wire [1:0] replay_tile_o,
    output wire [6:0] replay_start_feature_o,
    output wire [6:0] replay_feature_o,
    output wire [7:0] replay_beats_o,
    output wire       replay_first_o,
    output wire       replay_tile_last_o,
    output wire       scalar_valid_o,
    output wire       scalar_fire_o,
    output reg  [3:0] core_op_o,
    output wire       core_last_o,
    output wire       done_o
);
    localparam [4:0] RQ_STREAM    = 5'd0,  RQ_DRAIN     = 5'd1,
                     INIT_REQ     = 5'd2,  INIT_STREAM  = 5'd3,
                     INIT_DRAIN   = 5'd4,  ACC_REQ      = 5'd5,
                     ACC_STREAM   = 5'd6,  ACC_DRAIN    = 5'd7,
                     D_LAUNCH     = 5'd8,  D_DRAIN      = 5'd9,
                     RHO_LAUNCH   = 5'd10, RHO_DRAIN    = 5'd11,
                     NORM_REQ     = 5'd12, NORM_STREAM  = 5'd13,
                     NORM_DRAIN   = 5'd14, AFFINE_REQ   = 5'd15,
                     AFFINE_STREAM= 5'd16, AFFINE_DRAIN = 5'd17;
    reg [4:0] phase_r;
    reg [9:0] rq_count_r;
    reg [1:0] tile_r;
    reg [6:0] feature_r;
    reg rho_seen_r, tile_done_seen_r, commit_seen_r;

    wire run = rst_ni && active_i && !init_i;
    wire [9:0] rq_beats = {mt_i, 7'b0};
    wire final_tile = ({1'b0, tile_r} == mt_i - 3'd1);
    wire affine_phase = (phase_r == AFFINE_STREAM || phase_r == AFFINE_DRAIN);
    wire tile_complete = tile_done_seen_r || affine_tile_done_i;
    wire command_complete = commit_seen_r || commit_done_i;

    assign rq_en_o = run && phase_r == RQ_STREAM;
    assign rq_fire_o = rq_en_o && rq_valid_i;
    assign rq_last_o = rq_fire_o && (rq_count_r == rq_beats - 10'd1);
    assign replay_start_o = run && (phase_r == INIT_REQ || phase_r == ACC_REQ ||
                                   phase_r == NORM_REQ || phase_r == AFFINE_REQ);
    assign replay_en_o = run && (phase_r == INIT_STREAM || phase_r == ACC_STREAM ||
                         phase_r == AFFINE_STREAM ||
                         (phase_r == NORM_STREAM && rho_match_valid_i));
    assign replay_fire_o = replay_en_o && replay_valid_i;
    assign replay_tile_o = tile_r;
    assign replay_start_feature_o = (core_op_o == `VFU_OP_LN_MOMENT_ACC) ? 7'd1 : 7'd0;
    assign replay_feature_o = feature_r;
    assign replay_beats_o = (core_op_o == `VFU_OP_LN_MOMENT_INIT) ? 8'd1 :
                           (core_op_o == `VFU_OP_LN_MOMENT_ACC)  ? 8'd127 : 8'd128;
    assign replay_first_o = replay_fire_o && feature_r == replay_start_feature_o;
    // Feature127 is a tile boundary; INIT's sole feature0 is not tile-final.
    assign replay_tile_last_o = replay_fire_o && feature_r == 7'd127;
    assign scalar_valid_o = run && (phase_r == D_LAUNCH || phase_r == RHO_LAUNCH);
    assign scalar_fire_o = scalar_valid_o && scalar_ready_i;
    // Route this metadata with the accepted vector, never with a later counter.
    // Scalar/intermediate last markers must NOT enter ACT16 final commit.
    assign core_last_o = rq_last_o || scalar_fire_o ||
                        (replay_tile_last_o &&
                         (phase_r != AFFINE_STREAM || final_tile));
    assign done_o = run && phase_r == AFFINE_DRAIN && final_tile &&
                    tile_complete && command_complete;

    always @(*) begin
        core_op_o = `VFU_OP_RQ_RES;
        case (phase_r)
            INIT_REQ, INIT_STREAM, INIT_DRAIN: core_op_o = `VFU_OP_LN_MOMENT_INIT;
            ACC_REQ, ACC_STREAM, ACC_DRAIN: core_op_o = `VFU_OP_LN_MOMENT_ACC;
            D_LAUNCH, D_DRAIN: core_op_o = `VFU_OP_LN_D;
            RHO_LAUNCH, RHO_DRAIN: core_op_o = `VFU_OP_LN_RSQRT;
            NORM_REQ, NORM_STREAM, NORM_DRAIN: core_op_o = `VFU_OP_LN_NORM;
            AFFINE_REQ, AFFINE_STREAM, AFFINE_DRAIN: core_op_o = `VFU_OP_LN_AFFINE;
            default: core_op_o = `VFU_OP_RQ_RES;
        endcase
    end

    always @(posedge clk_i) begin
        if (!rst_ni || init_i) begin
            phase_r <= RQ_STREAM;
            rq_count_r <= 10'd0;
            tile_r <= 2'd0;
            feature_r <= 7'd0;
            rho_seen_r <= 1'b0;
            tile_done_seen_r <= 1'b0;
            commit_seen_r <= 1'b0;
        end else if (active_i) begin
            if (phase_r == RHO_DRAIN && rho_store_done_i) rho_seen_r <= 1'b1;
            if (affine_phase) begin
                if (affine_tile_done_i) tile_done_seen_r <= 1'b1;
                if (commit_done_i) commit_seen_r <= 1'b1;
            end
            case (phase_r)
                RQ_STREAM: if (rq_fire_o) begin
                    if (rq_last_o) phase_r <= RQ_DRAIN;
                    else rq_count_r <= rq_count_r + 10'd1;
                end
                RQ_DRAIN: if (z_store_done_i) phase_r <= INIT_REQ;
                INIT_REQ: phase_r <= INIT_STREAM;
                INIT_STREAM: if (replay_fire_o) phase_r <= INIT_DRAIN;
                INIT_DRAIN: if (init_retire_i) begin
                    phase_r <= ACC_REQ;
                    feature_r <= 7'd1;
                end
                ACC_REQ: phase_r <= ACC_STREAM;
                ACC_STREAM: if (replay_fire_o) begin
                    if (replay_tile_last_o) phase_r <= ACC_DRAIN;
                    else feature_r <= feature_r + 7'd1;
                end
                ACC_DRAIN: if (moment_capture_i) phase_r <= D_LAUNCH;
                D_LAUNCH: if (scalar_fire_o) phase_r <= D_DRAIN;
                D_DRAIN: if (d_result_valid_i) phase_r <= RHO_LAUNCH;
                RHO_LAUNCH: if (scalar_fire_o) phase_r <= RHO_DRAIN;
                RHO_DRAIN: if ((rho_seen_r || rho_store_done_i) && rho_match_valid_i) begin
                    phase_r <= NORM_REQ;
                    feature_r <= 7'd0;
                end
                NORM_REQ: phase_r <= NORM_STREAM;
                NORM_STREAM: if (replay_fire_o) begin
                    if (replay_tile_last_o) phase_r <= NORM_DRAIN;
                    else feature_r <= feature_r + 7'd1;
                end
                NORM_DRAIN: if (t_store_done_i) begin
                    phase_r <= AFFINE_REQ;
                    feature_r <= 7'd0;
                end
                AFFINE_REQ: phase_r <= AFFINE_STREAM;
                AFFINE_STREAM: if (replay_fire_o) begin
                    if (replay_tile_last_o) phase_r <= AFFINE_DRAIN;
                    else feature_r <= feature_r + 7'd1;
                end
                AFFINE_DRAIN: if (tile_complete && !final_tile) begin
                    phase_r <= INIT_REQ;
                    tile_r <= tile_r + 2'd1;
                    feature_r <= 7'd0;
                    rho_seen_r <= 1'b0;
                    tile_done_seen_r <= 1'b0;
                    commit_seen_r <= 1'b0;
                end
                default: phase_r <= RQ_STREAM;
            endcase
        end
    end

    // Caller errors are diagnosed in simulation; no abort/recovery protocol.
    // synthesis translate_off
    reg [2:0] checked_mt_r;
    always @(posedge clk_i) begin
        if (!rst_ni) checked_mt_r <= 3'd0;
        else if (init_i) begin
            if (mt_i < 1 || mt_i > 4) $fatal(1, "LN MT must be 1..4");
            checked_mt_r <= mt_i;
        end else if (run) begin
            if (mt_i !== checked_mt_r) $fatal(1, "LN MT changed while active");
            if (replay_valid_i && !replay_en_o)
                $fatal(1, "LN replay returned outside its prepared active burst");
            if (z_store_done_i && phase_r != RQ_DRAIN)
                $fatal(1, "LN z completion before ingress drain");
            if (init_retire_i && phase_r != INIT_DRAIN)
                $fatal(1, "LN INIT retirement outside INIT drain");
            if (moment_capture_i && phase_r != ACC_DRAIN)
                $fatal(1, "LN final S/Q capture outside Moment drain");
            if (d_result_valid_i && phase_r != D_DRAIN)
                $fatal(1, "LN D result outside D drain");
            if (rho_store_done_i && phase_r != RHO_DRAIN)
                $fatal(1, "LN rho capture outside RSQRT drain");
            if (t_store_done_i && phase_r != NORM_DRAIN)
                $fatal(1, "LN T completion outside NORM drain");
            if ((affine_tile_done_i || commit_done_i) &&
                !(phase_r == AFFINE_DRAIN || (phase_r == AFFINE_STREAM && replay_tile_last_o)))
                $fatal(1, "LN architectural completion before final tile launch");
            if (commit_done_i && !final_tile)
                $fatal(1, "LN command commit_done on nonfinal tile");
            if ((phase_r == NORM_REQ || phase_r == NORM_STREAM || phase_r == NORM_DRAIN) &&
                !rho_match_valid_i) $fatal(1, "LN rho lost before NORM drained");
        end
    end
    // synthesis translate_on
endmodule
