`timescale 1ns/1ps
`include "vfu_external_ops.vh"
`include "vfu_internal_op_defs.vh"

// Runnable wiring example for the common dispatcher and real child controllers.
// Arithmetic, memory and MAX/R payloads are boundary models, not a VFU datapath.
module tb_vfu_ln_hierarchy_draft;
    reg clk_i = 1'b0;
    always #5 clk_i = ~clk_i;
    reg rst_ni = 1'b0;
    reg start_i = 1'b0;
    reg [2:0] op_i = `VFU_RQ;
    reg [9:0] m_i = 10'd16, n_i = 10'd1;
    reg [6:0] seq_len_i = 7'd2;
    reg layer_i = 1'b0, num_ln_i = 1'b0;
    reg signed [17:0] req_mult_i = 18'sd0;
    reg [5:0] req_shamt_i = 6'd0;
    reg transpose_i = 1'b0;
    reg [2:0] write_sel_i = 3'd0;
    reg [15:0] write_base_i = 16'd0, res_base_i = 16'd0, ln_base_i = 16'd0;
    wire softmax_done_i;
    wire ln_done_i;
    reg beat_valid_i = 1'b0, commit_done_i = 1'b0;

    wire cmd_accept_o, setup_o, busy_o, done_o;
    wire rq_gelu_init_o, rq_gelu_active_o;
    wire softmax_init_o, softmax_active_o, ln_init_o, ln_active_o;
    wire [2:0] op_o;
    wire [9:0] m_o, n_o;
    wire [6:0] seq_len_o;
    wire layer_o, num_ln_o;
    wire signed [17:0] req_mult_o;
    wire [5:0] req_shamt_o;
    wire transpose_o;
    wire [2:0] write_sel_o;
    wire [15:0] write_base_o, res_base_o, ln_base_o;
    wire [2:0] mt_o;
    wire [11:0] total_beats_o;
    wire [3:0] rq_gelu_core_op_o;
    wire rq_gelu_done, stream_en, last_beat;

    vfu_common_control_draft dut (
        .clk_i(clk_i), .rst_ni(rst_ni), .start_i(start_i),
        .op_i(op_i), .m_i(m_i), .n_i(n_i), .seq_len_i(seq_len_i),
        .layer_i(layer_i), .num_ln_i(num_ln_i),
        .req_mult_i(req_mult_i), .req_shamt_i(req_shamt_i),
        .transpose_i(transpose_i), .write_sel_i(write_sel_i),
        .write_base_i(write_base_i), .res_base_i(res_base_i), .ln_base_i(ln_base_i),
        .rq_gelu_done_i(rq_gelu_done), .softmax_done_i(softmax_done_i), .ln_done_i(ln_done_i),
        .cmd_accept_o(cmd_accept_o), .setup_o(setup_o), .busy_o(busy_o), .done_o(done_o),
        .rq_gelu_init_o(rq_gelu_init_o), .rq_gelu_active_o(rq_gelu_active_o),
        .softmax_init_o(softmax_init_o), .softmax_active_o(softmax_active_o),
        .ln_init_o(ln_init_o), .ln_active_o(ln_active_o),
        .op_o(op_o), .m_o(m_o), .n_o(n_o), .seq_len_o(seq_len_o),
        .layer_o(layer_o), .num_ln_o(num_ln_o),
        .req_mult_o(req_mult_o), .req_shamt_o(req_shamt_o),
        .transpose_o(transpose_o), .write_sel_o(write_sel_o),
        .write_base_o(write_base_o), .res_base_o(res_base_o), .ln_base_o(ln_base_o),
        .mt_o(mt_o), .total_beats_o(total_beats_o), .rq_gelu_core_op_o(rq_gelu_core_op_o)
    );

    vfu_rq_gelu_control owner_rq_gelu (
        .clk_i(clk_i), .rstn_i(rst_ni),
        .init_i(rq_gelu_init_o), .active_i(rq_gelu_active_o),
        .beat_valid_i(beat_valid_i), .commit_done_i(commit_done_i),
        .total_beats_i(total_beats_o),
        .rq_gelu_done_o(rq_gelu_done), .stream_en_o(stream_en), .last_beat_o(last_beat)
    );

    reg score_valid = 0, max_store_done = 0;
    reg [1:0] score_tile = 0, max_store_tile = 0;
    reg [5:0] score_key = 0;
    reg replay_valid = 0, rowsum_valid = 0, e_tile_done = 0;
    reg recip_ready = 0, recip_store_done = 0;
    reg nl_valid = 0, context_commit_done = 0;
    reg [1:0] nl_tile = 0;
    wire score_en, score_fire, rowmax_clear, rowmax_last, score_key_valid;
    wire replay_start, replay_en, replay_key_valid, replay_first;
    wire replay_tile_last, replay_command_last, recip_valid;
    wire [1:0] replay_tile;
    wire [5:0] replay_key;
    wire nl_en, nl_fire, nl_last, state_clear;
    wire [3:0] softmax_core_op;

    // Model only slot lifetime/type. These integers are NOT computed MAX/R.
    integer slot_type [0:3]; // 0=invalid, 1=MAX, 2=R.
    integer slot_marker [0:3];
    integer j;
    integer clear_count = 0, max_count = 0, r_count = 0;
    integer rq_launch_count = 0, qexp_launch_count = 0;
    integer recip_launch_count = 0, nl_launch_count = 0;
    integer command_count = 0, done_count = 0;
    wire nl_r_valid = (slot_type[nl_tile] == 2);

    vfu_softmax_control_draft softmax (
        .clk_i(clk_i), .rst_ni(rst_ni),
        .init_i(softmax_init_o), .active_i(softmax_active_o), .op_i(op_o),
        .mt_i(mt_o), .seq_len_i(seq_len_o), .keys_i(m_o[6:0]),
        .score_valid_i(score_valid), .score_tile_i(score_tile), .score_key_i(score_key),
        .max_store_done_i(max_store_done), .max_store_tile_i(max_store_tile),
        .replay_valid_i(replay_valid), .rowsum_valid_i(rowsum_valid),
        .e_tile_done_i(e_tile_done), .recip_ready_i(recip_ready),
        .recip_store_done_i(recip_store_done),
        .nl_valid_i(nl_valid), .nl_tile_i(nl_tile), .nl_r_valid_i(nl_r_valid),
        .commit_done_i(context_commit_done),
        .score_en_o(score_en), .score_fire_o(score_fire),
        .rowmax_clear_o(rowmax_clear), .rowmax_last_o(rowmax_last),
        .score_key_valid_o(score_key_valid),
        .replay_start_o(replay_start), .replay_en_o(replay_en),
        .replay_tile_o(replay_tile), .replay_key_o(replay_key),
        .replay_key_valid_o(replay_key_valid), .replay_first_o(replay_first),
        .replay_tile_last_o(replay_tile_last), .replay_command_last_o(replay_command_last),
        .recip_valid_o(recip_valid), .nl_en_o(nl_en), .nl_fire_o(nl_fire),
        .nl_last_o(nl_last), .core_op_o(softmax_core_op),
        .state_clear_o(state_clear), .done_o(softmax_done_i)
    );

    // Example of selected control routing, not a second operation sequencer.
    // RQ/GELU mapping is static; Softmax selects QEXP/RECIP/CONTEXT by phase.
    wire rq_launch = rq_gelu_active_o && stream_en && beat_valid_i;
    wire exp_launch = softmax_active_o && replay_en && replay_valid;
    wire recip_launch = softmax_active_o && recip_valid && recip_ready;
    wire context_launch = softmax_active_o && nl_fire;
    wire core_launch = rq_launch || exp_launch || recip_launch || context_launch || ln_launch;
    wire [3:0] selected_core_op = rq_gelu_active_o ? rq_gelu_core_op_o :
                                        ln_active_o ? ln_core_op : softmax_core_op;

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            for (j = 0; j < 4; j = j + 1) begin
                slot_type[j] <= 0;
                slot_marker[j] <= 0;
            end
        end else begin
            if (state_clear) begin
                clear_count = clear_count + 1;
                for (j = 0; j < 4; j = j + 1) begin
                    slot_type[j] <= 0;
                    slot_marker[j] <= 0;
                end
            end
            if (max_store_done) begin
                slot_type[max_store_tile] <= 1;
                slot_marker[max_store_tile] <= 100 + max_store_tile;
                max_count = max_count + 1;
            end
            if (recip_store_done) begin
                slot_type[replay_tile] <= 2;
                slot_marker[replay_tile] <= 200 + replay_tile;
                r_count = r_count + 1;
            end
            if (done_o) done_count = done_count + 1;
            if (rq_launch) begin
                rq_launch_count = rq_launch_count + 1;
                if (selected_core_op !== ((op_o == `VFU_GELU) ? `VFU_OP_GELU : `VFU_OP_RQ))
                    $fatal(1, "wrong selected RQ/GELU opcode");
            end
            if (exp_launch) begin
                qexp_launch_count = qexp_launch_count + 1;
                if (selected_core_op !== `VFU_OP_QEXP || slot_type[replay_tile] != 1)
                    $fatal(1, "QEXP must consume MAX with QEXP opcode");
            end
            if (recip_launch) begin
                recip_launch_count = recip_launch_count + 1;
                if (selected_core_op !== `VFU_OP_SM_RECIP_RAW || slot_type[replay_tile] != 1)
                    $fatal(1, "reciprocal must retain old MAX until R write");
            end
            if (context_launch) begin
                nl_launch_count = nl_launch_count + 1;
                if (selected_core_op !== `VFU_OP_SM_CONTEXT || slot_type[nl_tile] != 2 ||
                    slot_marker[nl_tile] != 200 + nl_tile)
                    $fatal(1, "SM_NL must use matching retained R");
            end
            if ((core_launch || score_fire) && !busy_o)
                $fatal(1, "operation launched while common controller idle");
        end
    end

    task automatic start_command(input [2:0] op,
                                 input [9:0] tokens,
                                 input [9:0] features,
                                 input [6:0] original_tokens);
        reg [2:0] selected;
        begin
            @(negedge clk_i);
            if (busy_o !== 0) $fatal(1, "start requires idle");
            op_i = op; m_i = tokens; n_i = features; seq_len_i = original_tokens;
            req_mult_i = -18'sd73; req_shamt_i = 6'd7;
            write_base_i = 16'h8123; write_sel_i = 3'd4;
            start_i = 1;
            selected = (op == `VFU_RQ || op == `VFU_GELU) ? 3'b100 :
                       (op == `VFU_LN) ? 3'b001 : 3'b010;
            #1;
            if (cmd_accept_o !== 1) $fatal(1, "accept missing");
            @(posedge clk_i); #1;
            if (setup_o !== 1 || busy_o !== 1 || done_o !== 0 ||
                {rq_gelu_init_o, softmax_init_o, ln_init_o} !== selected ||
                {rq_gelu_active_o, softmax_active_o, ln_active_o} !== 3'b000)
                $fatal(1, "SETUP init/active contract violated");
            if (op == `VFU_SM_NL && state_clear !== 0)
                $fatal(1, "SM_NL init attempted to clear persistent R");
            @(negedge clk_i);
            start_i = 0;
            op_i = 3'b111; m_i = 0; n_i = 0; seq_len_i = 0;
            req_mult_i = 18'sd51; req_shamt_i = 6'd33;
            @(posedge clk_i); #1;
            if ({rq_gelu_active_o, softmax_active_o, ln_active_o} !== selected ||
                setup_o !== 0 || done_o !== 0 || op_o !== op || m_o !== tokens ||
                n_o !== features || seq_len_o !== original_tokens ||
                req_mult_o !== -18'sd73 || req_shamt_o !== 6'd7)
                $fatal(1, "CORE_RUN or captured configuration mismatch");
            command_count = command_count + 1;
        end
    endtask

    task automatic complete_command;
        begin
            #1;
            if (done_o !== 1 || busy_o !== 1)
                $fatal(1, "selected completion not forwarded in CORE_RUN");
            @(posedge clk_i); #1;
            if (done_o !== 0 || busy_o !== 0 ||
                {rq_gelu_active_o, softmax_active_o, ln_active_o} !== 0)
                $fatal(1, "common controller did not return directly to idle");
        end
    endtask

    task automatic check_r_persistent;
        begin
            if (clear_count != 1 || slot_type[0] != 2 || slot_type[1] != 2 ||
                slot_marker[0] != 200 || slot_marker[1] != 201)
                $fatal(1, "R lost across command or idle boundary");
        end
    endtask

    task automatic run_one_rq(input [2:0] op);
        begin
            start_command(op, 16, 1, 2);
            @(negedge clk_i); beat_valid_i = 1;
            #1;
            if (last_beat !== 1 || core_launch !== 1) $fatal(1, "one-beat RQ/GELU launch missing");
            @(posedge clk_i); #1;
            @(negedge clk_i); beat_valid_i = 0;
            repeat (3) begin
                @(posedge clk_i); #1;
                if (done_o !== 0 || stream_en !== 0 || rq_gelu_active_o !== 1)
                    $fatal(1, "RQ/GELU did not wait for commit");
            end
            @(negedge clk_i); commit_done_i = 1;
            complete_command();
            commit_done_i = 0;
        end
    endtask


    // Real LN control/helper; the following pipeline models only returned events.
    reg ln_rq_valid = 0, ln_replay_valid = 0, ln_scalar_ready = 0;
    reg [1:0] ln_input_tile = 0;
    reg [6:0] ln_input_feature = 0;
    reg z_stored = 0, init_retired = 0, moment_captured = 0;
    reg d_valid = 0, t_stored = 0, affine_tile_done = 0, ln_commit_done = 0;
    wire ln_rq_en, ln_rq_fire, ln_rq_last;
    wire ln_replay_start, ln_replay_en, ln_replay_fire, ln_replay_first;
    wire ln_replay_tile_last, ln_scalar_valid, ln_scalar_fire, ln_core_last;
    wire [1:0] ln_tile;
    wire [6:0] ln_start_feature, ln_feature;
    wire [7:0] ln_beats;
    wire [3:0] ln_core_op;
    wire [127:0] rho;
    wire rho_match, rho_stored;
    wire ln_launch = ln_active_o && (ln_rq_fire || ln_replay_fire || ln_scalar_fire);

    vfu_ln_control_draft ln (
        .clk_i(clk_i), .rst_ni(rst_ni), .init_i(ln_init_o), .active_i(ln_active_o),
        .mt_i(mt_o), .rq_valid_i(ln_rq_valid), .z_store_done_i(z_stored),
        .replay_valid_i(ln_replay_valid), .init_retire_i(init_retired),
        .moment_capture_i(moment_captured), .scalar_ready_i(ln_scalar_ready),
        .d_result_valid_i(d_valid), .rho_store_done_i(rho_stored),
        .rho_match_valid_i(rho_match), .t_store_done_i(t_stored),
        .affine_tile_done_i(affine_tile_done), .commit_done_i(ln_commit_done),
        .rq_en_o(ln_rq_en), .rq_fire_o(ln_rq_fire), .rq_last_o(ln_rq_last),
        .replay_start_o(ln_replay_start), .replay_en_o(ln_replay_en),
        .replay_fire_o(ln_replay_fire), .replay_tile_o(ln_tile),
        .replay_start_feature_o(ln_start_feature), .replay_feature_o(ln_feature),
        .replay_beats_o(ln_beats), .replay_first_o(ln_replay_first),
        .replay_tile_last_o(ln_replay_tile_last), .scalar_valid_o(ln_scalar_valid),
        .scalar_fire_o(ln_scalar_fire), .core_op_o(ln_core_op),
        .core_last_o(ln_core_last), .done_o(ln_done_i)
    );

    localparam LAT = 3; // Positive response latency; not a measured CORE latency.
    reg [LAT-1:0] result_valid = 0, result_last = 0, result_tile_last = 0;
    reg [3:0] result_op [0:LAT-1];
    reg [1:0] result_tile [0:LAT-1];
    reg [6:0] result_feature [0:LAT-1];
    reg [511:0] result_data [0:LAT-1];
    // Synthetic unrelated S3 results exercise capture qualification too.
    reg foreign_valid = 0;
    reg [3:0] foreign_op = `VFU_OP_SM_RECIP_RAW;
    reg [511:0] foreign_data = {16{32'h87654321}};
    wire model_retire = result_valid[LAT-1] || foreign_valid;
    wire [3:0] model_retire_op = result_valid[LAT-1] ? result_op[LAT-1] : foreign_op;
    wire [511:0] model_retire_data = result_valid[LAT-1] ? result_data[LAT-1] : foreign_data;
    wire [1:0] model_retire_tile = result_valid[LAT-1] ? result_tile[LAT-1] : 2'd0;
    wire rho_capture = ln_active_o && model_retire && model_retire_op == `VFU_OP_LN_RSQRT;
    vfu_ln_rho16_draft rho_store (
        .clk_i(clk_i), .rst_ni(rst_ni), .clear_i(ln_init_o),
        .capture_i(rho_capture), .capture_tile_i(model_retire_tile),
        .capture_data_i(model_retire_data), .rd_tile_i(ln_tile),
        .rho_o(rho), .match_valid_o(rho_match), .stored_o(rho_stored)
    );

    // Marker memory is deliberately not a numeric LN model or an S-pad instance.
    // Its illustrative address is tile*128+feature; type 1=z, type 2=T.
    integer scratch_type [0:511];
    integer scratch_marker [0:511];
    reg norm_read [0:511];
    integer sq_tile = -1, d_tile = -1;
    reg sq_valid = 0;
    reg [511:0] held_core_result = 0;
    integer ln_count [0:11];
    integer z_write_count = 0, t_write_count = 0, affine_write_count = 0;
    integer rho_capture_count = 0, burst_count = 0;
    integer pipe_index, lane, addr_read, addr_write;
    integer commit_delay = 0;
    reg [511:0] launch_marker;
    reg [127:0] last_rho = 0;
    reg expect_rho_hold = 0;

    function automatic [7:0] rho_marker(input integer tile, input integer lane_index);
        rho_marker = 8'd16 + tile*32 + lane_index;
    endfunction

    always @(*) begin
        launch_marker = {16{32'h13572468}};
        if (ln_core_op == `VFU_OP_LN_RSQRT)
            for (integer l = 0; l < 16; l = l + 1)
                launch_marker[l*32 +: 32] = {24'd0, rho_marker(ln_tile, l)};
    end

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            result_valid <= 0;
            for (pipe_index = 0; pipe_index < 512; pipe_index = pipe_index + 1) begin
                scratch_type[pipe_index] = 0;
                scratch_marker[pipe_index] = 0;
                norm_read[pipe_index] = 0;
            end
            for (pipe_index = 0; pipe_index < 12; pipe_index = pipe_index + 1)
                ln_count[pipe_index] = 0;
        end else begin
            z_stored <= 0; init_retired <= 0; moment_captured <= 0;
            d_valid <= 0; t_stored <= 0; affine_tile_done <= 0; ln_commit_done <= 0;
            if (commit_delay != 0) begin
                commit_delay = commit_delay - 1;
                if (commit_delay == 0) ln_commit_done <= 1;
            end
            for (pipe_index = LAT-1; pipe_index > 0; pipe_index = pipe_index - 1) begin
                result_valid[pipe_index] <= result_valid[pipe_index-1];
                result_last[pipe_index] <= result_last[pipe_index-1];
                result_tile_last[pipe_index] <= result_tile_last[pipe_index-1];
                result_op[pipe_index] <= result_op[pipe_index-1];
                result_tile[pipe_index] <= result_tile[pipe_index-1];
                result_feature[pipe_index] <= result_feature[pipe_index-1];
                result_data[pipe_index] <= result_data[pipe_index-1];
            end
            result_valid[0] <= ln_launch;
            result_op[0] <= selected_core_op;
            result_tile[0] <= ln_rq_fire ? ln_input_tile : ln_tile;
            result_feature[0] <= ln_rq_fire ? ln_input_feature : ln_feature;
            result_last[0] <= ln_core_last;
            result_tile_last[0] <= ln_replay_tile_last;
            result_data[0] <= launch_marker;
            if (ln_replay_start) burst_count = burst_count + 1;
            if (ln_launch) begin
                ln_count[selected_core_op] = ln_count[selected_core_op] + 1;
                if (op_o !== `VFU_LN || !busy_o || !ln_active_o || rq_gelu_active_o || softmax_active_o)
                    $fatal(1, "LN launch ownership/mux mismatch");
                if (ln_replay_fire) begin
                    addr_read = ln_tile*128 + ln_feature;
                    if (scratch_type[addr_read] !== ((ln_core_op == `VFU_OP_LN_AFFINE) ? 2 : 1) ||
                        scratch_marker[addr_read] !== addr_read+1)
                        $fatal(1, "LN replay wrong data lifetime/tile/feature");
                    if (ln_core_op == `VFU_OP_LN_NORM) begin
                        if (norm_read[addr_read]) $fatal(1, "z location read twice during NORM");
                        norm_read[addr_read] = 1;
                        if (!sq_valid || sq_tile != ln_tile || !rho_match)
                            $fatal(1, "NORM missing retained S or matching rho");
                        for (lane = 0; lane < 16; lane = lane + 1)
                            if (rho[lane*8 +: 8] !== rho_marker(ln_tile, lane))
                                $fatal(1, "rho packing/lifetime mismatch");
                    end
                end
                if (ln_scalar_fire && ln_core_op == `VFU_OP_LN_D &&
                    (!sq_valid || sq_tile != ln_tile)) $fatal(1, "D before final S/Q capture");
                if (ln_scalar_fire && ln_core_op == `VFU_OP_LN_RSQRT &&
                    (d_tile != ln_tile || held_core_result !== {16{32'h13572468}}))
                    $fatal(1, "RSQRT did not consume held D result");
            end
            if (model_retire) held_core_result <= model_retire_data;
            if (rho_capture) rho_capture_count = rho_capture_count + 1;
            if (result_valid[LAT-1]) begin
                addr_write = result_tile[LAT-1]*128 + result_feature[LAT-1];
                case (result_op[LAT-1])
                    `VFU_OP_RQ_RES: begin
                        if (scratch_type[addr_write] != 0) $fatal(1, "duplicate z write");
                        scratch_type[addr_write] = 1;
                        scratch_marker[addr_write] = addr_write+1;
                        z_write_count = z_write_count + 1;
                        if (result_last[LAT-1]) z_stored <= 1;
                    end
                    `VFU_OP_LN_MOMENT_INIT: begin
                        if (result_last[LAT-1]) $fatal(1, "INIT must not capture S/Q");
                        init_retired <= 1;
                    end
                    `VFU_OP_LN_MOMENT_ACC: if (result_last[LAT-1]) begin
                        sq_valid = 1; sq_tile = result_tile[LAT-1]; moment_captured <= 1;
                    end
                    `VFU_OP_LN_D: begin d_tile = result_tile[LAT-1]; d_valid <= 1; end
                    `VFU_OP_LN_NORM: begin
                        if (!norm_read[addr_write] || scratch_type[addr_write] != 1)
                            $fatal(1, "T overwrite preceded z read");
                        if (ln_replay_fire && addr_read == addr_write)
                            $fatal(1, "forbidden same-cycle scratch read/write collision");
                        scratch_type[addr_write] = 2;
                        t_write_count = t_write_count + 1;
                        if (result_tile_last[LAT-1]) t_stored <= 1;
                    end
                    `VFU_OP_LN_AFFINE: begin
                        affine_write_count = affine_write_count + 1;
                        if (result_tile_last[LAT-1]) affine_tile_done <= 1;
                        if (result_last[LAT-1]) commit_delay = 3;
                    end
                endcase
            end
            // Every unrelated command/result must leave the helper unchanged.
            if (expect_rho_hold && !rho_capture && !ln_init_o && rho_match && rho !== last_rho)
                $fatal(1, "unqualified result changed rho");
            if (rho_match) begin last_rho = rho; expect_rho_hold = 1; end
            if (ln_init_o || rho_capture || !rho_match) expect_rho_hold = 0;
        end
    end

    task automatic replay_ln_burst(input [3:0] expected_op,
                                   input integer tile,
                                   input integer first_feature,
                                   input integer count);
        begin
            wait (ln_replay_start === 1);
            #1;
            if (ln_core_op !== expected_op || ln_tile !== tile[1:0] ||
                ln_start_feature !== first_feature[6:0] || ln_beats !== count[7:0])
                $fatal(1, "LN burst request mismatch");
            @(posedge clk_i); #1;
            for (integer f = first_feature; f < first_feature+count; f = f + 1) begin
                @(negedge clk_i); ln_replay_valid = 1;
                #1;
                if (!ln_replay_fire || ln_feature !== f[6:0] ||
                    ln_replay_first !== (f == first_feature) ||
                    ln_replay_tile_last !== (f == 127) ||
                    ln_core_last !== ((f == 127) && (expected_op != `VFU_OP_LN_AFFINE || tile == 1)))
                    $fatal(1, "LN replay first/tile-last/command-last mismatch");
                @(posedge clk_i); #1;
                if (f % 19 == 4) begin
                    @(negedge clk_i); ln_replay_valid = 0;
                    @(posedge clk_i); #1;
                end
            end
            @(negedge clk_i); ln_replay_valid = 0;
        end
    endtask

    task automatic launch_ln_scalar(input [3:0] expected_op);
        begin
            wait (ln_scalar_valid === 1);
            repeat (2) @(posedge clk_i);
            @(negedge clk_i); ln_scalar_ready = 1;
            #1;
            if (ln_core_op !== expected_op || !ln_scalar_fire || !ln_core_last)
                $fatal(1, "LN singleton request mismatch");
            @(posedge clk_i); #1;
            @(negedge clk_i); ln_scalar_ready = 0;
        end
    endtask

    task automatic run_ln;
        begin
            start_command(`VFU_LN, 32, 128, 17);
            if (rho_match || rho_stored || state_clear) $fatal(1, "LN init clear ownership mismatch");
            check_r_persistent();
            for (integer b = 0; b < 256; b = b + 1) begin
                @(negedge clk_i);
                ln_rq_valid = 1; ln_input_tile = b / 128;
                ln_input_feature = ((b % 128) / 16)*16 + 15 - (b % 16);
                #1;
                if (!ln_rq_fire || ln_rq_last !== (b == 255) || selected_core_op !== `VFU_OP_RQ_RES)
                    $fatal(1, "tagged LN ingress mismatch");
                @(posedge clk_i); #1;
            end
            @(negedge clk_i); ln_rq_valid = 0;
            for (integer tile = 0; tile < 2; tile = tile + 1) begin
                replay_ln_burst(`VFU_OP_LN_MOMENT_INIT, tile, 0, 1);
                replay_ln_burst(`VFU_OP_LN_MOMENT_ACC, tile, 1, 127);
                launch_ln_scalar(`VFU_OP_LN_D);
                launch_ln_scalar(`VFU_OP_LN_RSQRT);
                replay_ln_burst(`VFU_OP_LN_NORM, tile, 0, 128);
                replay_ln_burst(`VFU_OP_LN_AFFINE, tile, 0, 128);
                check_r_persistent();
            end
            wait (done_o === 1); complete_command();
            if (ln_count[`VFU_OP_RQ_RES] != 256 || ln_count[`VFU_OP_LN_MOMENT_INIT] != 2 ||
                ln_count[`VFU_OP_LN_MOMENT_ACC] != 254 || ln_count[`VFU_OP_LN_D] != 2 ||
                ln_count[`VFU_OP_LN_RSQRT] != 2 || ln_count[`VFU_OP_LN_NORM] != 256 ||
                ln_count[`VFU_OP_LN_AFFINE] != 256 || z_write_count != 256 || t_write_count != 256 ||
                affine_write_count != 256 || rho_capture_count != 2 || burst_count != 8)
                $fatal(1, "LN hierarchy coverage mismatch");
            check_r_persistent();
            // Same opcode without LN ownership, and a foreign opcode while idle,
            // cannot overwrite rho even if an unrelated result bus is active.
            @(negedge clk_i); foreign_valid = 1; foreign_op = `VFU_OP_LN_RSQRT;
            repeat (2) @(posedge clk_i);
            @(negedge clk_i); foreign_op = `VFU_OP_SM_RECIP_RAW;
            repeat (2) @(posedge clk_i);
            @(negedge clk_i); foreign_valid = 0;
            if (rho_capture_count != 2) $fatal(1, "rho capture missing selected-LN qualification");
        end
    endtask

    integer t, k;
    initial begin
        repeat (3) @(posedge clk_i);
        @(negedge clk_i); rst_ni = 1;
        start_command(`VFU_SM, 32, 32, 17);
        for (t = 0; t < 2; t = t + 1) begin
            for (k = 0; k < 32; k = k + 1) begin
                @(negedge clk_i);
                score_valid = 1; score_tile = t;
                score_key = (k / 16) * 16 + 15 - (k % 16);
                #1;
                if (score_fire !== 1 || rowmax_clear !== (k == 0) ||
                    rowmax_last !== (k == 31) || score_key_valid !== (score_key < 17))
                    $fatal(1, "wrong tagged score acceptance");
                @(posedge clk_i); #1;
            end
            @(negedge clk_i);
            score_valid = 0; max_store_done = 1; max_store_tile = t;
            @(posedge clk_i); #1;
            @(negedge clk_i); max_store_done = 0;
        end
        for (t = 0; t < 2; t = t + 1) begin
            // Observe the one-cycle replay request before sending its returns.
            if (replay_start !== 1 || replay_tile !== t[1:0])
                $fatal(1, "missing whole-tile replay request");
            for (k = 0; k < 32; k = k + 1) begin
                @(negedge clk_i); replay_valid = 1;
                #1;
                if (replay_en !== 1 || replay_key !== k[5:0] ||
                    replay_key_valid !== (k < 17) || replay_first !== (k == 0) ||
                    replay_tile_last !== (k == 31) ||
                    replay_command_last !== (t == 1 && k == 31))
                    $fatal(1, "wrong replay boundary metadata");
                @(posedge clk_i); #1;
            end
            @(negedge clk_i); replay_valid = 0; rowsum_valid = 1;
            @(posedge clk_i); #1;
            @(negedge clk_i); rowsum_valid = 0;
            repeat (2) @(posedge clk_i);
            #1;
            if (recip_valid !== 0 || done_o !== 0) $fatal(1, "reciprocal before E completion");
            @(negedge clk_i); e_tile_done = 1;
            @(posedge clk_i); #1;
            @(negedge clk_i); e_tile_done = 0;
            if (recip_valid !== 1) $fatal(1, "missing reciprocal request");
            repeat (2) @(posedge clk_i);
            @(negedge clk_i); recip_ready = 1;
            @(posedge clk_i); #1;
            @(negedge clk_i); recip_ready = 0;
            repeat (2) begin
                @(posedge clk_i); #1;
                if (done_o !== 0 || busy_o !== 1 || softmax_active_o !== 1)
                    $fatal(1, "SM completed before final R write");
            end
            @(negedge clk_i); recip_store_done = 1;
            if (t == 1) complete_command();
            else begin @(posedge clk_i); #1; end
            @(negedge clk_i); recip_store_done = 0;
        end
        check_r_persistent();
        repeat (3) @(posedge clk_i);
        #1; check_r_persistent();
        run_ln(); // A real LN child must preserve the independent Softmax R slots.
        check_r_persistent();
        start_command(`VFU_SM_NL, 32, 64, 17);
        check_r_persistent();
        for (k = 0; k < 128; k = k + 1) begin
            @(negedge clk_i); nl_valid = 1; nl_tile = k / 64;
            #1;
            if (nl_fire !== 1 || nl_last !== (k == 127))
                $fatal(1, "SM_NL wrong numerator acceptance/last");
            @(posedge clk_i); #1;
        end
        @(negedge clk_i); nl_valid = 0;
        repeat (3) begin
            @(posedge clk_i); #1;
            if (done_o !== 0 || busy_o !== 1 || softmax_active_o !== 1 || nl_en !== 0)
                $fatal(1, "SM_NL did not wait in drain for context commit");
        end
        @(negedge clk_i); context_commit_done = 1;
        complete_command();
        context_commit_done = 0;
        check_r_persistent();
        run_one_rq(`VFU_RQ);
        run_one_rq(`VFU_GELU);
        check_r_persistent();
        if (command_count != 5 || done_count != 5 || rq_launch_count != 2 ||
            qexp_launch_count != 64 || recip_launch_count != 2 ||
            nl_launch_count != 128 || max_count != 2 || r_count != 2)
            $fatal(1, "hierarchy event counts mismatch");
        $display("[PASS] LN hierarchy: SM -> LN -> SM_NL -> RQ -> GELU; LN MT2 1028 launches, 8 replay bursts, 256 z/T/final writes, 2 rho captures; persistent R and selected dispatch");
        $finish;
    end

    initial begin
        #2000000;
        $fatal(1, "LN hierarchy test timeout");
    end
endmodule
