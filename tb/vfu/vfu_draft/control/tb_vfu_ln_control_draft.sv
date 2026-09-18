`timescale 1ns/1ps
`include "vfu_internal_op_defs.vh"

// Control-boundary model ONLY. No native CORE arithmetic, coefficients, S-pad
// RAM or ACT16 implementation. A fixed three-edge result queue preserves tags;
// completion events are deliberately delayed further by the stimulus tasks.
// Scratch type markers exercise a proposed in-place z -> T overlay, not a
// frozen physical address layout. Each vector stands for all sixteen lanes.
module tb_vfu_ln_control_draft;
    reg clk = 0;
    always #5 clk = !clk;
    reg rst_n = 0, init = 0, active = 0;
    reg [2:0] mt = 1;
    reg rq_valid = 0, z_done = 0, replay_valid = 0, init_retire = 0;
    reg moment_capture = 0, scalar_ready = 0, d_valid = 0, t_done = 0;
    reg tile_done = 0, commit_done = 0;
    wire rq_en, rq_fire, rq_last, replay_start, replay_en, replay_fire;
    wire [1:0] tile;
    wire [6:0] start_feature, feature;
    wire [7:0] beats;
    wire replay_first, replay_last, scalar_valid, scalar_fire, core_last, done;
    wire [3:0] core_op;
    reg capture = 0, extra_clear = 0;
    reg [1:0] capture_tile = 0;
    reg [511:0] capture_data = 0;
    wire [127:0] rho;
    wire rho_match, rho_stored;

    vfu_ln_control_draft dut (
        .clk_i(clk), .rst_ni(rst_n), .init_i(init), .active_i(active), .mt_i(mt),
        .rq_valid_i(rq_valid), .z_store_done_i(z_done), .replay_valid_i(replay_valid),
        .init_retire_i(init_retire), .moment_capture_i(moment_capture),
        .scalar_ready_i(scalar_ready), .d_result_valid_i(d_valid),
        .rho_store_done_i(rho_stored), .rho_match_valid_i(rho_match),
        .t_store_done_i(t_done), .affine_tile_done_i(tile_done), .commit_done_i(commit_done),
        .rq_en_o(rq_en), .rq_fire_o(rq_fire), .rq_last_o(rq_last),
        .replay_start_o(replay_start), .replay_en_o(replay_en), .replay_fire_o(replay_fire),
        .replay_tile_o(tile), .replay_start_feature_o(start_feature),
        .replay_feature_o(feature), .replay_beats_o(beats),
        .replay_first_o(replay_first), .replay_tile_last_o(replay_last),
        .scalar_valid_o(scalar_valid), .scalar_fire_o(scalar_fire),
        .core_op_o(core_op), .core_last_o(core_last), .done_o(done)
    );
    vfu_ln_rho16_draft u_rho (
        .clk_i(clk), .rst_ni(rst_n), .clear_i(init || extra_clear),
        .capture_i(capture), .capture_tile_i(capture_tile), .capture_data_i(capture_data),
        .rd_tile_i(tile), .rho_o(rho), .match_valid_o(rho_match), .stored_o(rho_stored)
    );

    integer cycle = 0, commands = 0, all_launches = 0;
    integer ingress_count = 0, z_writes = 0, norm_writes = 0, final_writes = 0;
    integer moment_results = 0, launches = 0, replay_requests = 0;
    integer init_launches = 0, acc_launches = 0, d_launches = 0, rho_launches = 0;
    integer norm_launches = 0, affine_launches = 0;
    integer seed = 1, rng_seed, junk;
    integer scratch_type [0:511]; // 0=empty, 1=z, 2=T.
    integer scratch_stamp [0:511];
    integer last_read_cycle [0:511];
    reg [511:0] core_result = 0; // Held result boundary, analogous to data_o.
    reg [31:0] sq_stamp = 0;     // Model of CORE's held S/Q, not a DUT bank.
    reg [1:0] ingress_tile = 0;
    reg [6:0] ingress_feature = 0;
    reg pipe_valid [0:2];
    reg [3:0] pipe_op [0:2];
    integer pipe_addr [0:2];
    reg [1:0] pipe_tile [0:2];
    reg pipe_last [0:2];
    reg pipe_tile_last [0:2];
    integer p, lane, addr_now, addr_write;
    integer command_salt = 0;
    reg [127:0] expected_rho;
    string negative_case;

    function [511:0] rho_vector(input integer tile_num);
        integer l;
        begin
            rho_vector = 0;
            for (l = 0; l < 16; l = l + 1)
                rho_vector[l*32 +: 32] = (tile_num * 37 + l * 7 + 1) & 255;
        end
    endfunction
    function [127:0] rho_bytes(input integer tile_num);
        integer l;
        begin
            rho_bytes = 0;
            for (l = 0; l < 16; l = l + 1)
                rho_bytes[l*8 +: 8] = (tile_num * 37 + l * 7 + 1) & 255;
        end
    endfunction

    // A result queue decouples actual input acceptance from writes/retirement.
    // It asserts no same-cycle equal-address replay-read/result-write. NORM
    // overwrites only z already consumed at least one earlier edge; later
    // features and other tiles keep their original z until their own turn.
    always @(posedge clk) begin
        cycle = cycle + 1;
        if (!rst_n || init) begin
            for (p = 0; p < 3; p = p + 1) pipe_valid[p] <= 0;
        end else begin
            for (p = 2; p > 0; p = p - 1) begin
                pipe_valid[p] <= pipe_valid[p-1];
                pipe_op[p] <= pipe_op[p-1];
                pipe_addr[p] <= pipe_addr[p-1];
                pipe_tile[p] <= pipe_tile[p-1];
                pipe_last[p] <= pipe_last[p-1];
                pipe_tile_last[p] <= pipe_tile_last[p-1];
            end
            pipe_valid[0] <= rq_fire || replay_fire || scalar_fire;
            pipe_op[0] <= core_op;
            pipe_last[0] <= core_last;
            pipe_tile_last[0] <= replay_last;
            pipe_tile[0] <= rq_fire ? ingress_tile : tile;
            addr_now = rq_fire ? ingress_tile*128 + ingress_feature : tile*128 + feature;
            pipe_addr[0] <= addr_now;

            if (rq_fire || replay_fire || scalar_fire) begin
                if (!active || init || !rst_n) $fatal(1, "Launch outside active command");
                launches = launches + 1;
                all_launches = all_launches + 1;
            end
            if (replay_start) replay_requests = replay_requests + 1;
            if (rq_fire) begin
                if (ingress_tile >= mt || ingress_feature > 127)
                    $fatal(1, "Ingress tag outside geometry");
                ingress_count = ingress_count + 1;
            end
            if (replay_fire) begin
                if (addr_now < 0 || addr_now >= 512) $fatal(1, "Scratch address overflow");
                if (scratch_type[addr_now] !== (core_op == `VFU_OP_LN_AFFINE ? 2 : 1))
                    $fatal(1, "Wrong scratch lifetime at addr=%0d op=%0h", addr_now, core_op);
                if (scratch_stamp[addr_now] !== command_salt + addr_now +
                    (core_op == `VFU_OP_LN_AFFINE ? 10000 : 0))
                    $fatal(1, "Data/coordinate misalignment");
                last_read_cycle[addr_now] = cycle;
                case (core_op)
                    `VFU_OP_LN_MOMENT_INIT: init_launches = init_launches + 1;
                    `VFU_OP_LN_MOMENT_ACC: acc_launches = acc_launches + 1;
                    `VFU_OP_LN_NORM: begin
                        norm_launches = norm_launches + 1;
                        if (rho !== rho_bytes(tile) || sq_stamp !== command_salt + tile)
                            $fatal(1, "NORM lost held rho or S/Q");
                    end
                    `VFU_OP_LN_AFFINE: affine_launches = affine_launches + 1;
                    default: $fatal(1, "Unexpected replay micro-op");
                endcase
            end
            if (scalar_fire) begin
                if (core_op == `VFU_OP_LN_D) begin
                    d_launches = d_launches + 1;
                    if (sq_stamp !== command_salt + tile)
                        $fatal(1, "D launched before correct S/Q");
                end else if (core_op == `VFU_OP_LN_RSQRT) begin
                    rho_launches = rho_launches + 1;
                    if (core_result[31:0] !== 32'hd0000000 + tile)
                        $fatal(1, "RSQRT lost held D result");
                end else $fatal(1, "Unexpected scalar operation");
            end
            if (pipe_valid[2]) begin
                // No next micro-op may launch before previous results drain.
                if (core_op !== pipe_op[2]) $fatal(1, "Op changed with in-flight result");
                addr_write = pipe_addr[2];
                case (pipe_op[2])
                    `VFU_OP_RQ_RES: begin
                        if (scratch_type[addr_write] !== 0) $fatal(1, "Duplicate z coordinate");
                        scratch_type[addr_write] = 1;
                        scratch_stamp[addr_write] = command_salt + addr_write;
                        z_writes = z_writes + 1;
                        core_result = {16{32'h10000000 + addr_write}};
                    end
                    `VFU_OP_LN_MOMENT_INIT: begin
                        if (pipe_last[2]) $fatal(1, "INIT incorrectly captures Moment");
                        core_result = 0;
                    end
                    `VFU_OP_LN_MOMENT_ACC: begin
                        if (pipe_last[2]) begin
                            if (addr_write % 128 != 127) $fatal(1, "Early Moment capture");
                            sq_stamp = command_salt + pipe_tile[2];
                            moment_results = moment_results + 1;
                        end
                        core_result = 0;
                    end
                    `VFU_OP_LN_D: core_result = {16{32'hd0000000 + {30'd0, pipe_tile[2]}}};
                    `VFU_OP_LN_RSQRT: core_result = rho_vector(pipe_tile[2]);
                    `VFU_OP_LN_NORM: begin
                        if (scratch_type[addr_write] !== 1 || last_read_cycle[addr_write] >= cycle)
                            $fatal(1, "NORM overwrite before reading z");
                        if (replay_fire && addr_now == addr_write)
                            $fatal(1, "Forbidden same-address scratch read/write collision");
                        scratch_type[addr_write] = 2;
                        scratch_stamp[addr_write] = command_salt + addr_write + 10000;
                        norm_writes = norm_writes + 1;
                        core_result = {16{32'hac000000 + addr_write}};
                    end
                    `VFU_OP_LN_AFFINE: begin
                        final_writes = final_writes + 1;
                        if (pipe_last[2] !== (pipe_tile_last[2] && pipe_tile[2] == mt-1))
                            $fatal(1, "Tile/command last mixed at architectural writer");
                        core_result = {16{32'h7f}};
                    end
                endcase
            end
        end
    end

    task step;
        begin @(posedge clk); #1; @(negedge clk); #1; end
    endtask
    task drain(input [3:0] expected_op, input integer clocks);
        integer c;
        begin
            for (c = 0; c < clocks; c = c + 1) begin
                #1;
                if (core_op !== expected_op || rq_en || replay_en || replay_start ||
                    scalar_valid || core_last || done)
                    $fatal(1, "Drain leaked launch or changed op=%0h expected=%0h", core_op, expected_op);
                step;
            end
        end
    endtask
    task replay_pass(input [3:0] op, input integer first, input integer length,
                     input integer tile_num);
        integer f;
        reg expected_last;
        begin
            #1;
            if (!replay_start || replay_en || core_op !== op || tile !== tile_num[1:0] ||
                start_feature !== first[6:0] || beats !== length[7:0])
                $fatal(1, "Incorrect replay request op=%0h tile=%0d start=%0d beats=%0d", core_op, tile, start_feature, beats);
            step;
            for (f = first; f < first + length; f = f + 1) begin
                if ((f + seed + tile_num) % 7 == 0) begin
                    replay_valid = 0;
                    #1;
                    if (!replay_en || replay_fire || replay_first || replay_last || core_last ||
                        feature !== f[6:0]) $fatal(1, "Bubble advanced replay");
                    step;
                end
                replay_valid = 1;
                expected_last = (f == 127) && (op != `VFU_OP_LN_AFFINE || tile_num == mt-1);
                #1;
                if (!replay_en || !replay_fire || replay_start || feature !== f[6:0] ||
                    core_op !== op || tile !== tile_num[1:0] ||
                    replay_first !== (f == first) || replay_last !== (f == 127) ||
                    core_last !== expected_last || done)
                    $fatal(1, "Replay metadata/fire mismatch op=%0h feature=%0d", op, f);
                if (op == `VFU_OP_LN_NORM && rho !== rho_bytes(tile_num))
                    $fatal(1, "Rho data changed during NORM");
                step;
                replay_valid = 0;
            end
        end
    endtask
    task scalar_launch(input [3:0] op, input integer wait_clocks);
        integer c;
        begin
            scalar_ready = 0;
            for (c = 0; c < wait_clocks; c = c + 1) begin
                #1;
                if (!scalar_valid || scalar_fire || core_last || core_op !== op || done)
                    $fatal(1, "Scalar launch ignored preparation wait");
                step;
            end
            scalar_ready = 1;
            #1;
            if (!scalar_valid || !scalar_fire || !core_last) $fatal(1, "Missing scalar launch");
            step;
            scalar_ready = 0;
        end
    endtask

    task run_ln(input integer tile_count, input integer join_order);
        integer b, t, a;
        begin
            active = 0;
            mt = tile_count;
            command_salt = (commands + 1)*100000;
            ingress_count = 0; z_writes = 0; norm_writes = 0; final_writes = 0;
            launches = 0; replay_requests = 0; moment_results = 0;
            init_launches = 0; acc_launches = 0; d_launches = 0; rho_launches = 0;
            norm_launches = 0; affine_launches = 0;
            for (a = 0; a < 512; a = a + 1) begin
                scratch_type[a] = 0; scratch_stamp[a] = 0; last_read_cycle[a] = -1;
            end
            init = 1;
            step;
            if (rho_match || rho_stored || rho !== 128'd0 || rq_en || replay_en || scalar_valid)
                $fatal(1, "Init did not invalidate rho/suppress launches");
            init = 0;
            repeat (2) begin
                #1;
                if (rq_en || replay_start || scalar_valid || done) $fatal(1, "Inactive controller leaked work");
                step;
            end
            active = 1;
            for (b = 0; b < tile_count*128; b = b + 1) begin
                // Preserve true producer coordinates: reverse feature15..0
                // inside each 16-feature block, while tile order ascends.
                ingress_tile = b / 128;
                ingress_feature = ((b % 128)/16)*16 + 15 - (b % 16);
                if ((b + seed) % 9 == 0) begin
                    rq_valid = 0;
                    #1;
                    if (!rq_en || rq_fire || rq_last) $fatal(1, "Ingress bubble counted");
                    step;
                end
                rq_valid = 1;
                #1;
                if (!rq_fire || core_op !== `VFU_OP_RQ_RES ||
                    rq_last !== (b == tile_count*128-1) || core_last !== rq_last)
                    $fatal(1, "Ingress count/last mismatch at %0d", b);
                step;
                rq_valid = 0;
            end
            drain(`VFU_OP_RQ_RES, 5 + seed%3);
            if (z_writes != tile_count*128) $fatal(1, "z completion ahead of writes");
            for (a = 0; a < tile_count*128; a = a + 1)
                if (scratch_type[a] !== 1) $fatal(1, "Incomplete full-command z storage");
            z_done = 1; step; z_done = 0;
            for (t = 0; t < tile_count; t = t + 1) begin
                replay_pass(`VFU_OP_LN_MOMENT_INIT, 0, 1, t);
                drain(`VFU_OP_LN_MOMENT_INIT, 5);
                if (moment_results != t) $fatal(1, "INIT produced final S/Q capture");
                init_retire = 1; step; init_retire = 0;
                replay_pass(`VFU_OP_LN_MOMENT_ACC, 1, 127, t);
                drain(`VFU_OP_LN_MOMENT_ACC, 6);
                if (moment_results != t+1) $fatal(1, "Missing final S/Q result");
                moment_capture = 1; step; moment_capture = 0;
                scalar_launch(`VFU_OP_LN_D, 3);
                drain(`VFU_OP_LN_D, 5);
                d_valid = 1; step; d_valid = 0;
                scalar_launch(`VFU_OP_LN_RSQRT, 4);
                drain(`VFU_OP_LN_RSQRT, 5);
                capture_data = core_result;
                if (capture_data !== rho_vector(t)) $fatal(1, "Wrong RSQRT boundary payload");
                // A stored result with a different tile cannot start NORM.
                capture_tile = t ^ 1;
                capture = 1; step; capture = 0;
                if (rho_match || rho !== 128'd0 || !rho_stored)
                    $fatal(1, "Rho mismatched-tag read not filtered");
                drain(`VFU_OP_LN_RSQRT, 3);
                capture_tile = t;
                capture = 1; step; capture = 0;
                if (!rho_match || !rho_stored || rho !== rho_bytes(t))
                    $fatal(1, "Rho lane packing/capture pulse mismatch");
                step; // Controller sees registered stored_o here.
                if (rho_stored) $fatal(1, "Rho stored pulse repeated");
                replay_pass(`VFU_OP_LN_NORM, 0, 128, t);
                drain(`VFU_OP_LN_NORM, 5);
                if (norm_writes != (t+1)*128 || rho !== rho_bytes(t))
                    $fatal(1, "T/rho lifetime incomplete");
                t_done = 1; step; t_done = 0;
                replay_pass(`VFU_OP_LN_AFFINE, 0, 128, t);
                drain(`VFU_OP_LN_AFFINE, 5);
                if (final_writes != (t+1)*128) $fatal(1, "Affine done ahead of final write");
                if (t != tile_count-1) begin
                    tile_done = 1; step; tile_done = 0;
                end else begin
                    if (join_order == 0) begin
                        tile_done = 1;
                        #1; if (done) $fatal(1, "done before command completion");
                        step; tile_done = 0;
                        drain(`VFU_OP_LN_AFFINE, 3);
                        commit_done = 1;
                    end else if (join_order == 1) begin
                        commit_done = 1;
                        #1; if (done) $fatal(1, "done before tile completion");
                        step; commit_done = 0;
                        drain(`VFU_OP_LN_AFFINE, 4);
                        tile_done = 1;
                    end else begin
                        tile_done = 1; commit_done = 1;
                    end
                    #1;
                    if (!done || rq_en || replay_en || scalar_valid)
                        $fatal(1, "Missing joined completion / final drain leaked launch");
                    step;
                    active = 0; tile_done = 0; commit_done = 0;
                    #1; if (done) $fatal(1, "Inactive done not gated");
                end
            end
            if (launches != tile_count*514 || ingress_count != tile_count*128 ||
                replay_requests != tile_count*4 || init_launches != tile_count ||
                acc_launches != tile_count*127 || d_launches != tile_count ||
                rho_launches != tile_count || norm_launches != tile_count*128 ||
                affine_launches != tile_count*128)
                $fatal(1, "Wrong exact launch counts MT=%0d count=%0d", tile_count, launches);
            commands = commands + 1;
            $display("LN command %0d: MT=%0d, launches=%0d, join_order=%0d", commands, tile_count, launches, join_order);
            step;
        end
    endtask

    initial begin
        if (!$value$plusargs("SEED=%d", seed)) seed = 1;
        rng_seed = seed; junk = $urandom(rng_seed);
        repeat (2) step;
        rst_n = 1;
        if ($value$plusargs("NEG=%s", negative_case)) begin
            if (negative_case == "bad_mt") begin
                mt = 0; init = 1; step;
            end else if (negative_case == "bad_clear") begin
                extra_clear = 1; capture = 1; step;
            end else if (negative_case == "bad_u8") begin
                capture_data[31:8] = 1; capture = 1; step;
            end else if (negative_case == "bad_event") begin
                init = 1; step; init = 0; active = 1; moment_capture = 1; step;
            end else if (negative_case == "bad_replay") begin
                init = 1; step; init = 0; active = 1; replay_valid = 1; step;
            end else $fatal(1, "Unknown negative test");
            $fatal(1, "Negative test was not diagnosed");
        end
        run_ln(1, 0);
        run_ln(4, 1);
        run_ln(2, 2);
        run_ln(3, 0);
        run_ln(4, 2);
        // Reset invalidates existing retained rho even without a new command.
        rst_n = 0; step;
        if (rho_match || rho_stored || rho !== 128'd0) $fatal(1, "Reset retained rho validity");
        $display("[PASS] LN controller/rho boundary model: commands=%0d launches=%0d seed=%0d", commands, all_launches, seed);
        $finish;
    end
    initial begin #3000000; $fatal(1, "LN test timeout"); end
endmodule
