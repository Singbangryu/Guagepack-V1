`timescale 1ns/1ps
`include "vfu_external_ops.vh"
`include "vfu_internal_op_defs.vh"

// Boundary-model test only: no numeric CORE, S-pad, ACT16 or state64 instance.
module tb_vfu_softmax_control_draft;
    reg clk_i = 0;
    always #5 clk_i = ~clk_i;
    reg rst_ni = 0, init_i = 0, active_i = 0;
    reg [2:0] op_i = `VFU_SM, mt_i = 1;
    reg [6:0] seq_len_i = 2, keys_i = 16;
    reg score_valid_i = 0;
    reg [1:0] score_tile_i = 0;
    reg [5:0] score_key_i = 0;
    reg max_store_done_i = 0;
    reg [1:0] max_store_tile_i = 0;
    reg replay_valid_i = 0, rowsum_valid_i = 0, e_tile_done_i = 0;
    reg recip_ready_i = 0, recip_store_done_i = 0;
    reg nl_valid_i = 0;
    reg [1:0] nl_tile_i = 0;
    reg nl_r_valid_i = 0, commit_done_i = 0;
    wire score_en_o, score_fire_o, rowmax_clear_o, rowmax_last_o;
    wire score_key_valid_o, replay_start_o, replay_en_o;
    wire [1:0] replay_tile_o;
    wire [5:0] replay_key_o;
    wire replay_key_valid_o, replay_first_o, replay_tile_last_o;
    wire replay_command_last_o, recip_valid_o, nl_en_o, nl_fire_o, nl_last_o;
    wire [3:0] core_op_o;
    wire state_clear_o, done_o;
    vfu_softmax_control_draft dut (.*);

    integer total_score = 0, total_replay = 0, total_nl = 0;
    integer total_recip = 0, commands = 0, completion_count = 0;
    integer expected_score, expected_replay, expected_nl, expected_recip;
    integer starts, case_idx, mode;
    integer lengths [0:11];
    reg [3:0] mock_valid = 0, mock_r_type = 0;
    reg [3:0] r_before_idle;
    reg [63:0] observed_keys [0:3];
    integer monitor_tile, monitor_key;

    task tick;
        begin @(posedge clk_i); #1; @(negedge clk_i); end
    endtask
    task idle_cycles(input integer cycles);
        integer i;
        begin for (i = 0; i < cycles; i = i + 1) tick(); end
    endtask
    task check(input bit condition, input string message);
        begin if (!condition) $fatal(1, "%s at t=%0t", message, $time); end
    endtask
    task clear_inputs;
        begin
            score_valid_i = 0; max_store_done_i = 0; replay_valid_i = 0;
            rowsum_valid_i = 0; e_tile_done_i = 0; recip_ready_i = 0;
            recip_store_done_i = 0; nl_valid_i = 0; nl_r_valid_i = 0;
            commit_done_i = 0;
        end
    endtask
    task initialize(input bit nl);
        begin
            active_i = 0;
            clear_inputs();
            op_i = nl ? `VFU_SM_NL : `VFU_SM;
            init_i = 1;
            #1;
            check(state_clear_o === !nl, "Only SM init clears row state");
            check(!score_fire_o && !nl_fire_o && !replay_en_o, "Init disables input");
            tick();
            init_i = 0; active_i = 1;
            expected_score = 0; expected_replay = 0; expected_nl = 0;
            expected_recip = 0; starts = 0;
            commands = commands + 1;
        end
    endtask

    // The mock state has persistent tile valid/type bits only; no data/numerics.
    // This checker is external to the DUT and observes its public event contract.
    always @(posedge clk_i) begin
        if (!rst_ni) begin
            mock_valid <= 0; mock_r_type <= 0;
        end else begin
            if (state_clear_o) begin
                mock_valid <= 0; mock_r_type <= 0;
            end else begin
                if (max_store_done_i && active_i) begin
                    mock_valid[max_store_tile_i] <= 1;
                    mock_r_type[max_store_tile_i] <= 0;
                end
                if (recip_store_done_i && active_i) begin
                    mock_valid[replay_tile_o] <= 1;
                    mock_r_type[replay_tile_o] <= 1;
                end
            end
            if (!active_i || init_i) begin
                check(!score_fire_o && !replay_start_o && !replay_first_o &&
                      !replay_tile_last_o && !recip_valid_o && !nl_fire_o && !done_o,
                      "Inactive controller emitted a request/event");
            end
            if (score_fire_o) begin
                monitor_tile = expected_score / keys_i;
                monitor_key = expected_score % keys_i;
                check(score_tile_i == monitor_tile, "Score true tile mismatch");
                check(rowmax_clear_o === (monitor_key == 0), "Rowmax clear mismatch");
                check(rowmax_last_o === (monitor_key == keys_i - 1), "Rowmax last mismatch");
                check(score_key_valid_o === (score_key_i < seq_len_i), "Key mask mismatch");
                if (monitor_key == 0) observed_keys[monitor_tile] = 0;
                check(!observed_keys[monitor_tile][score_key_i], "Repeated input key");
                observed_keys[monitor_tile][score_key_i] = 1;
                if (monitor_key == keys_i - 1)
                    check(observed_keys[monitor_tile] == (64'hffffffffffffffff >> (64 - keys_i)),
                          "Not all keys ingested");
                expected_score = expected_score + 1; total_score = total_score + 1;
            end
            if (replay_start_o) begin
                check(mock_valid[replay_tile_o] && !mock_r_type[replay_tile_o],
                      "QEXP needs current tile MAX, not R or invalid state");
                check(replay_tile_o == starts, "Replay request tile order mismatch");
                starts = starts + 1;
            end
            if (replay_en_o && replay_valid_i) begin
                monitor_tile = expected_replay / keys_i;
                monitor_key = expected_replay % keys_i;
                check(replay_tile_o == monitor_tile && replay_key_o == monitor_key,
                      "Replay coordinate mismatch");
                check(replay_first_o === (monitor_key == 0), "Replay first mismatch");
                check(replay_tile_last_o === (monitor_key == keys_i - 1), "Tile last mismatch");
                check(replay_command_last_o === (expected_replay == mt_i*keys_i - 1),
                      "Head last must differ from tile last");
                check(replay_key_valid_o === (monitor_key < seq_len_i), "Replay key mask mismatch");
                check(core_op_o === `VFU_OP_QEXP, "Wrong QEXP opcode");
                expected_replay = expected_replay + 1; total_replay = total_replay + 1;
            end
            if (recip_valid_o && recip_ready_i) begin
                check(core_op_o === `VFU_OP_SM_RECIP_RAW, "Wrong reciprocal opcode");
                check(expected_replay == (expected_recip+1)*keys_i,
                      "Reciprocal before complete replay tile");
                expected_recip = expected_recip + 1; total_recip = total_recip + 1;
            end
            if (nl_fire_o) begin
                check(nl_r_valid_i && mock_valid[nl_tile_i] && mock_r_type[nl_tile_i],
                      "N consumed without matching retained R");
                check(core_op_o === `VFU_OP_SM_CONTEXT, "Wrong SM_NL opcode");
                check(nl_last_o === (expected_nl == mt_i*64 - 1), "SM_NL last mismatch");
                expected_nl = expected_nl + 1; total_nl = total_nl + 1;
            end
            if (done_o) begin
                completion_count = completion_count + 1;
                if (op_i == `VFU_SM)
                    check(expected_score == mt_i*keys_i && expected_replay == mt_i*keys_i &&
                          expected_recip == mt_i && starts == mt_i, "Premature SM completion");
                else check(expected_nl == mt_i*64, "Premature SM_NL completion");
            end
        end
    end

    task run_sm;
        integer t, k, order_key;
        begin
            initialize(0);
            for (t=0; t<mt_i; t=t+1) begin
                for (k=0; k<keys_i; k=k+1) begin
                    score_valid_i = 0;
                    if ((k + t + case_idx) % 7 == 0) begin
                        idle_cycles(2);
                        check(!rowmax_clear_o && !rowmax_last_o, "Bubble advanced rowmax");
                    end
                    order_key = (k/16)*16 + 15 - (k%16);
                    score_tile_i = t; score_key_i = order_key; score_valid_i = 1;
                    #1; check(score_en_o, "Stopped score input early");
                    tick();
                end
                score_valid_i = 0;
                // Earlier MAX results arrive during SCORE; final MAX is delayed.
                idle_cycles((t == mt_i-1) ? 4 : 1);
                check(!replay_en_o, "Replay started before all MAX stores");
                max_store_tile_i = t; max_store_done_i = 1;
                tick(); max_store_done_i = 0;
            end
            #1; check(!score_en_o, "Score input must stop after entire head");
            while (!replay_start_o) tick();
            for (t=0; t<mt_i; t=t+1) begin
                mode = (case_idx+t)%5;
                check(replay_start_o && replay_tile_o == t, "Missing single replay-start pulse");
                for (k=0; k<keys_i; k=k+1) begin
                    replay_valid_i = 0;
                    if ((k + case_idx) % 9 == 0) idle_cycles(1);
                    replay_valid_i = 1;
                    // Zero-delay boundary stress: latch event on final input edge.
                    if (k == keys_i-1 && mode == 3) rowsum_valid_i = 1;
                    if (k == keys_i-1 && mode == 4) e_tile_done_i = 1;
                    tick();
                    rowsum_valid_i = 0; e_tile_done_i = 0;
                end
                // Keep valid asserted briefly: the stopped phase must not consume it.
                check(!replay_en_o && !done_o, "Final input is not final completion");
                tick(); replay_valid_i = 0;
                check(core_op_o == `VFU_OP_QEXP, "QEXP opcode changed during drain");
                case (mode)
                    0: begin
                        rowsum_valid_i=1; tick(); rowsum_valid_i=0;
                        idle_cycles(3); check(!recip_valid_o, "L alone advanced phase");
                        e_tile_done_i=1; tick(); e_tile_done_i=0;
                    end
                    1: begin
                        e_tile_done_i=1; tick(); e_tile_done_i=0;
                        idle_cycles(2); check(!recip_valid_o, "E alone advanced phase");
                        rowsum_valid_i=1; tick(); rowsum_valid_i=0;
                    end
                    2: begin
                        rowsum_valid_i=1; e_tile_done_i=1; tick();
                        rowsum_valid_i=0; e_tile_done_i=0;
                    end
                    3: begin
                        idle_cycles(2); check(!recip_valid_o, "Lost E dependency");
                        e_tile_done_i=1; tick(); e_tile_done_i=0;
                    end
                    4: begin
                        idle_cycles(2); check(!recip_valid_o, "Lost L dependency");
                        rowsum_valid_i=1; tick(); rowsum_valid_i=0;
                    end
                endcase
                while (!recip_valid_o) tick();
                idle_cycles(3);
                check(expected_recip == t && !done_o, "Reciprocal request consumed without ready");
                recip_ready_i=1; tick();
                check(!recip_valid_o, "Reciprocal issued more than once");
                idle_cycles(2); recip_ready_i=0;
                check(core_op_o == `VFU_OP_SM_RECIP_RAW && !done_o,
                      "Reciprocal drain must hold opcode and wait for R store");
                recip_store_done_i=1; #1;
                check(done_o === (t==mt_i-1), "SM completion must wait final R store");
                tick(); recip_store_done_i=0;
            end
            active_i=0;
            r_before_idle = mock_r_type;
            idle_cycles(3);
            check(mock_r_type == r_before_idle && mock_valid == (4'b1111 >> (4-mt_i)),
                  "SM idle destroyed retained R");
        end
    endtask

    task run_nl;
        integer n;
        begin
            initialize(1);
            check(mock_r_type == r_before_idle, "SM_NL init cleared retained R");
            for (n=0; n<mt_i*64; n=n+1) begin
                nl_tile_i = n/64;
                nl_valid_i = 1; nl_r_valid_i = 0;
                if (n%11 == 0) begin
                    #1; check(!nl_en_o && !nl_fire_o && !nl_last_o, "Missing R must block N");
                    idle_cycles(2);
                end
                nl_r_valid_i = 1;
                if (n%13 == 0) begin
                    nl_valid_i=0; tick(); nl_valid_i=1;
                end
                tick();
            end
            check(!nl_en_o && !done_o, "Final N launch must wait final context write");
            idle_cycles(4);
            check(!done_o && core_op_o == `VFU_OP_SM_CONTEXT, "SM_NL drain mismatch");
            commit_done_i=1; #1; check(done_o, "Missing final context completion");
            tick(); commit_done_i=0; active_i=0; clear_inputs();
            idle_cycles(1);
            check(mock_r_type == r_before_idle, "SM_NL completion cleared R");
        end
    endtask

    initial begin
        lengths[0]=2; lengths[1]=15; lengths[2]=16; lengths[3]=17;
        lengths[4]=31; lengths[5]=32; lengths[6]=33; lengths[7]=47;
        lengths[8]=48; lengths[9]=49; lengths[10]=63; lengths[11]=64;
        idle_cycles(2); rst_ni=1; tick();
        for (case_idx=0; case_idx<12; case_idx=case_idx+1) begin
            seq_len_i = lengths[case_idx];
            mt_i = (lengths[case_idx]+15)/16; keys_i = mt_i*16;
            run_sm(); run_nl();
        end
        check(commands == 24 && completion_count == 24, "Missing or duplicate command completion");
        $display("[PASS] softmax draft: %0d commands, score=%0d replay=%0d reciprocals=%0d N=%0d",
                 commands, total_score, total_replay, total_recip, total_nl);
        $finish;
    end
    initial begin #1000000; $fatal(1, "TB watchdog timeout"); end
endmodule
