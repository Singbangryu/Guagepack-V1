`timescale 1ns/1ps
`include "vfu_external_ops.vh"
`include "vfu_internal_op_defs.vh"

// DRAFT: request/event controller only. No memory, CORE or row-state datapath.
// SM stores a full head, then replays query tiles in ascending order. Ingress
// tiles must be contiguous and ascending; keys within each tile may be permuted.
// score_fire is BOTH accepted scratch write and accepted rowmax input. Scratch
// writes must have completed by that edge; no pending write queue is modeled.
// All *_done/valid completion events below refer to the current operation/tile
// and are one-cycle, post-write events. Metadata must travel with the datapath.
//
// init works with active=0. SM init invalidates MAX/R metadata; SM_NL init does
// not. The caller preserves R, head ownership and geometry between SM and SM_NL.
// nl_r_valid means the current nl_tile FF read is valid AND has type R. It does
// not mean merely that some R exists. State64 read latency is combinational.
module vfu_softmax_control_draft (
    input  wire       clk_i,
    input  wire       rst_ni,
    input  wire       init_i,
    input  wire       active_i,
    input  wire [2:0] op_i,             // Command-static VFU_SM or VFU_SM_NL.
    input  wire [2:0] mt_i,             // 1..4 query tiles.
    input  wire [6:0] seq_len_i,        // 2..64 original, unpadded key count.
    input  wire [6:0] keys_i,           // 16*mt: padded keys, including 64.

    input  wire       score_valid_i,
    input  wire [1:0] score_tile_i,
    input  wire [5:0] score_key_i,
    input  wire       max_store_done_i,
    input  wire [1:0] max_store_tile_i,

    input  wire       replay_valid_i,   // Returned score, ready for CORE launch.
    input  wire       rowsum_valid_i,   // Final L captured/held until reciprocal.
    input  wire       e_tile_done_i,    // Last E write of THIS tile completed.
    input  wire       recip_ready_i,    // Adapter can launch reciprocal vector.
    input  wire       recip_store_done_i, // Current tile MAX->R write completed.

    input  wire       nl_valid_i,
    input  wire [1:0] nl_tile_i,
    input  wire       nl_r_valid_i,
    input  wire       commit_done_i,    // Final context write completed (SM_NL).

    output wire       score_en_o,
    output wire       score_fire_o,
    output wire       rowmax_clear_o,
    output wire       rowmax_last_o,
    output wire       score_key_valid_o,
    output wire       replay_start_o,  // Whole-tile request; guaranteed accepted.
    output wire       replay_en_o,
    output wire [1:0] replay_tile_o,
    output wire [5:0] replay_key_o,     // Replay adapter returns ascending keys.
    output wire       replay_key_valid_o,
    output wire       replay_first_o,
    output wire       replay_tile_last_o,
    output wire       replay_command_last_o,
    output wire       recip_valid_o,   // CORE launch = this request AND ready.
    output wire       nl_en_o,
    output wire       nl_fire_o,
    output wire       nl_last_o,
    output reg  [3:0] core_op_o,       // Use only with the selected launch event.
    output wire       state_clear_o,
    output wire       done_o
);
    localparam [3:0] SM_SCORE       = 4'd0;
    localparam [3:0] SM_MAX_DRAIN   = 4'd1;
    localparam [3:0] SM_EXP         = 4'd2;
    localparam [3:0] SM_EXP_DRAIN   = 4'd3;
    localparam [3:0] SM_RECIP       = 4'd4;
    localparam [3:0] SM_RECIP_DRAIN = 4'd5;
    localparam [3:0] NL_STREAM      = 4'd6;
    localparam [3:0] NL_DRAIN       = 4'd7;

    reg [3:0] phase_r;
    reg [6:0] score_count_r;
    reg [1:0] score_tile_r;
    reg [3:0] max_seen_r;
    reg [1:0] replay_tile_r;
    reg [6:0] replay_count_r;
    reg       replay_start_r;
    reg       rowsum_seen_r;
    reg       e_seen_r;
    reg [8:0] nl_count_r;

    wire run = rst_ni && active_i && !init_i;
    wire replay_fire = replay_en_o && replay_valid_i;
    wire recip_fire = recip_valid_o && recip_ready_i;
    wire [8:0] nl_beats = {mt_i, 6'b0}; // MT*64, including 256.
    wire last_tile = ({1'b0, replay_tile_r} == mt_i - 3'd1);
    wire [3:0] max_required = 4'b1111 >> (3'd4 - mt_i);
    wire [3:0] max_with_event = max_seen_r |
                         (max_store_done_i ? (4'b0001 << max_store_tile_i) : 4'b0);

    assign score_en_o = run && (phase_r == SM_SCORE);
    assign score_fire_o = score_en_o && score_valid_i;
    assign rowmax_clear_o = score_fire_o && (score_count_r == 7'd0);
    assign rowmax_last_o = score_fire_o && (score_count_r == keys_i - 7'd1);
    assign score_key_valid_o = ({1'b0, score_key_i} < seq_len_i);

    assign replay_start_o = run && (phase_r == SM_EXP) && replay_start_r;
    assign replay_en_o = run && (phase_r == SM_EXP);
    assign replay_tile_o = replay_tile_r;
    assign replay_key_o = replay_count_r[5:0];
    assign replay_key_valid_o = (replay_count_r < seq_len_i);
    assign replay_first_o = replay_fire && (replay_count_r == 7'd0);
    assign replay_tile_last_o = replay_fire && (replay_count_r == keys_i - 7'd1);
    // Tile-last must reach rowsum; command-last reaches the E writer descriptor.
    // One CORE last bit cannot substitute for these two different meanings.
    assign replay_command_last_o = replay_tile_last_o && last_tile;
    assign recip_valid_o = run && (phase_r == SM_RECIP);

    assign nl_en_o = run && (phase_r == NL_STREAM) && nl_r_valid_i;
    assign nl_fire_o = nl_en_o && nl_valid_i;
    assign nl_last_o = nl_fire_o && (nl_count_r == nl_beats - 9'd1);
    assign state_clear_o = rst_ni && init_i && (op_i == `VFU_SM);
    assign done_o = run && (((phase_r == SM_RECIP_DRAIN) && last_tile &&
                            recip_store_done_i) ||
                           ((phase_r == NL_DRAIN) && commit_done_i));

    always @(*) begin
        core_op_o = `VFU_OP_QEXP;
        case (phase_r)
            SM_RECIP, SM_RECIP_DRAIN: core_op_o = `VFU_OP_SM_RECIP_RAW;
            NL_STREAM, NL_DRAIN:      core_op_o = `VFU_OP_SM_CONTEXT;
            default:                 core_op_o = `VFU_OP_QEXP;
        endcase
    end

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            phase_r        <= SM_SCORE;
            score_count_r  <= 7'd0;
            score_tile_r   <= 2'd0;
            max_seen_r     <= 4'd0;
            replay_tile_r  <= 2'd0;
            replay_count_r <= 7'd0;
            replay_start_r <= 1'b0;
            rowsum_seen_r  <= 1'b0;
            e_seen_r       <= 1'b0;
            nl_count_r     <= 9'd0;
        end else if (init_i) begin
            phase_r        <= (op_i == `VFU_SM_NL) ? NL_STREAM : SM_SCORE;
            score_count_r  <= 7'd0;
            score_tile_r   <= 2'd0;
            max_seen_r     <= 4'd0;
            replay_tile_r  <= 2'd0;
            replay_count_r <= 7'd0;
            replay_start_r <= 1'b0;
            rowsum_seen_r  <= 1'b0;
            e_seen_r       <= 1'b0;
            nl_count_r     <= 9'd0;
        end else if (active_i) begin
            replay_start_r <= 1'b0;
            // MAX writes can complete while later score tiles are still arriving.
            if ((phase_r == SM_SCORE || phase_r == SM_MAX_DRAIN) && max_store_done_i)
                max_seen_r <= max_with_event;
            // Retain either completion if L and the final E write are separated.
            if (phase_r == SM_EXP || phase_r == SM_EXP_DRAIN) begin
                if (rowsum_valid_i) rowsum_seen_r <= 1'b1;
                if (e_tile_done_i)  e_seen_r      <= 1'b1;
            end
            case (phase_r)
                SM_SCORE: if (score_fire_o) begin
                    if (rowmax_last_o) begin
                        score_count_r <= 7'd0;
                        if ({1'b0, score_tile_r} == mt_i - 3'd1)
                            phase_r <= SM_MAX_DRAIN;
                        else
                            score_tile_r <= score_tile_r + 2'd1;
                    end else score_count_r <= score_count_r + 7'd1;
                end
                SM_MAX_DRAIN: if ((max_with_event & max_required) == max_required) begin
                    phase_r        <= SM_EXP;
                    replay_start_r <= 1'b1;
                end
                SM_EXP: if (replay_fire) begin
                    if (replay_tile_last_o) phase_r <= SM_EXP_DRAIN;
                    else replay_count_r <= replay_count_r + 7'd1;
                end
                SM_EXP_DRAIN: if ((rowsum_seen_r || rowsum_valid_i) &&
                                  (e_seen_r || e_tile_done_i))
                    phase_r <= SM_RECIP;
                SM_RECIP: if (recip_fire) phase_r <= SM_RECIP_DRAIN;
                SM_RECIP_DRAIN: if (recip_store_done_i && !last_tile) begin
                    phase_r        <= SM_EXP;
                    replay_tile_r  <= replay_tile_r + 2'd1;
                    replay_count_r <= 7'd0;
                    replay_start_r <= 1'b1;
                    rowsum_seen_r  <= 1'b0;
                    e_seen_r       <= 1'b0;
                end
                NL_STREAM: if (nl_fire_o) begin
                    if (nl_last_o) phase_r <= NL_DRAIN;
                    else nl_count_r <= nl_count_r + 9'd1;
                end
                NL_DRAIN: phase_r <= NL_DRAIN;
                default: phase_r <= SM_SCORE;
            endcase
        end
    end

    // Simulation-only contract checks: malformed schedules are caller errors,
    // not a request to add recovery states or silently drop padded beats.
    // synthesis translate_off
    reg [63:0] score_keys_seen_r;
    reg [3:0] score_tiles_done_r;
    reg [2:0] checked_op_r;
    reg [2:0] checked_mt_r;
    reg [6:0] checked_seq_r;
    reg [6:0] checked_keys_r;
    always @(posedge clk_i) begin
        if (!rst_ni) begin
            score_keys_seen_r <= 64'd0;
            score_tiles_done_r <= 4'd0;
            checked_op_r <= 3'd0;
            checked_mt_r <= 3'd0;
            checked_seq_r <= 7'd0;
            checked_keys_r <= 7'd0;
        end else if (init_i) begin
            if ((op_i != `VFU_SM) && (op_i != `VFU_SM_NL))
                $fatal(1, "Softmax init requires SM or SM_NL");
            if (mt_i < 1 || mt_i > 4 || keys_i != {mt_i, 4'b0} ||
                seq_len_i < 2 || seq_len_i > keys_i || seq_len_i <= keys_i - 16)
                $fatal(1, "Softmax geometry must use ceil16(seq_len), MT=1..4");
            score_keys_seen_r <= 64'd0;
            score_tiles_done_r <= 4'd0;
            checked_op_r <= op_i;
            checked_mt_r <= mt_i;
            checked_seq_r <= seq_len_i;
            checked_keys_r <= keys_i;
        end else if (active_i) begin
            if ({op_i, mt_i, seq_len_i, keys_i} !==
                {checked_op_r, checked_mt_r, checked_seq_r, checked_keys_r})
                $fatal(1, "Softmax command configuration changed while active");
            if (score_fire_o) begin
                if (score_tile_i != score_tile_r || {1'b0, score_key_i} >= keys_i)
                    $fatal(1, "Score tiles must be contiguous ascending, keys in range");
                if (score_keys_seen_r[score_key_i])
                    $fatal(1, "Duplicate score key within tile");
                if (rowmax_last_o) begin
                    score_keys_seen_r <= 64'd0;
                    score_tiles_done_r[score_tile_i] <= 1'b1;
                end else score_keys_seen_r[score_key_i] <= 1'b1;
            end
            if (max_store_done_i) begin
                if (!(phase_r == SM_SCORE || phase_r == SM_MAX_DRAIN) ||
                    {1'b0, max_store_tile_i} >= mt_i ||
                    !score_tiles_done_r[max_store_tile_i] || max_seen_r[max_store_tile_i])
                    $fatal(1, "MAX completion must be unique post-write for finished tile");
            end
            if (rowsum_valid_i || e_tile_done_i) begin
                if (!(phase_r == SM_EXP || phase_r == SM_EXP_DRAIN))
                    $fatal(1, "L/E completion outside current QEXP tile");
                if ((rowsum_valid_i && rowsum_seen_r) || (e_tile_done_i && e_seen_r))
                    $fatal(1, "Repeated L/E completion for one tile");
            end
            if (recip_store_done_i && phase_r != SM_RECIP_DRAIN)
                $fatal(1, "R completion before reciprocal request accepted");
            if (nl_fire_o && {1'b0, nl_tile_i} >= mt_i)
                $fatal(1, "SM_NL tile outside MT");
            if (commit_done_i && phase_r != NL_DRAIN)
                $fatal(1, "Context completion before final numerator launch");
        end
    end
    // synthesis translate_on
endmodule
