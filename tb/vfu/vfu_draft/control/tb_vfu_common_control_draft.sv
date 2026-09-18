`timescale 1ns/1ps
`include "vfu_external_ops.vh"
`include "vfu_internal_op_defs.vh"

// Dispatcher boundary test. RQ/GELU uses the unmodified Owner RTL;
// Softmax/LN done inputs below are explicit controller boundary stubs.
module tb_vfu_common_control_draft;
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
    reg softmax_done_i = 1'b0, ln_done_i = 1'b0;
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

    wire [107:0] captured_config = {
        op_o, m_o, n_o, seq_len_o, layer_o, num_ln_o, req_mult_o,
        req_shamt_o, transpose_o, write_sel_o, write_base_o, res_base_o, ln_base_o
    };
    reg [107:0] expected_config;
    integer expected_mt, expected_beats;
    integer command_count = 0;
    integer accepted_beats = 0;
    integer init_count = 0;
    integer completion_count = 0;
    integer negative_case = 0;
    reg check_hold = 1'b0;

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

    // Observe events at their accepting edges, before sequential updates.
    always @(posedge clk_i) begin
        if (rst_ni) begin
            if (stream_en && beat_valid_i) accepted_beats = accepted_beats + 1;
            if (rq_gelu_init_o || softmax_init_o || ln_init_o) init_count = init_count + 1;
            if (done_o) completion_count = completion_count + 1;
        end
    end

    task automatic check_descriptor;
        begin
            if (captured_config !== expected_config)
                $fatal(1, "captured descriptor changed or mismatched");
            if ((mt_o !== expected_mt[2:0]) || (total_beats_o !== expected_beats[11:0]))
                $fatal(1, "MT/total mismatch: MT=%0d total=%0d", mt_o, total_beats_o);
        end
    endtask

    always @(negedge clk_i) begin
        // Delay separates this checker from task-driven input mutation.
        #2;
        if (rst_ni && check_hold) check_descriptor();
    end

    task automatic launch_command(input [2:0] command_op,
                                  input integer tokens,
                                  input integer features,
                                  input integer original_tokens);
        reg [2:0] group_bits;
        begin
            @(negedge clk_i);
            if (busy_o !== 1'b0) $fatal(1, "launch requires idle");
            check_hold = 1'b0;
            op_i = command_op;
            m_i = tokens;
            n_i = features;
            seq_len_i = original_tokens;
            layer_i = command_count[0];
            num_ln_i = !command_count[0];
            req_mult_i = command_count[0] ? 18'sd131071 : -18'sd131072;
            req_shamt_i = command_count[0] ? 6'd63 : 6'd0;
            transpose_i = command_count[0];
            write_sel_i = command_count[2:0];
            write_base_i = 16'hf012 + command_count;
            res_base_i = 16'h8123 + command_count;
            ln_base_i = 16'hc234 + command_count;
            expected_config = {
                op_i, m_i, n_i, seq_len_i, layer_i, num_ln_i, req_mult_i,
                req_shamt_i, transpose_i, write_sel_i, write_base_i, res_base_i, ln_base_i
            };
            expected_mt = tokens / 16;
            expected_beats = expected_mt * features;
            group_bits = ((command_op == `VFU_RQ) || (command_op == `VFU_GELU)) ? 3'b100 :
                         ((command_op == `VFU_SM) || (command_op == `VFU_SM_NL)) ? 3'b010 : 3'b001;
            // An unselected child's stale done must never finish this command.
            softmax_done_i = !group_bits[1];
            ln_done_i = !group_bits[0];
            commit_done_i = 1'b0;
            beat_valid_i = 1'b1; // Availability even during SETUP must not launch.
            start_i = 1'b1;
            #1;
            if ((cmd_accept_o !== 1'b1) || (done_o !== 1'b0))
                $fatal(1, "bad idle acceptance");
            @(posedge clk_i); #1;
            check_hold = 1'b1;
            check_descriptor();
            if ((busy_o !== 1'b1) || (setup_o !== 1'b1) || (done_o !== 1'b0))
                $fatal(1, "bad SETUP outputs");
            if ({rq_gelu_init_o, softmax_init_o, ln_init_o} !== group_bits)
                $fatal(1, "SETUP must init selected child only");
            if ({rq_gelu_active_o, softmax_active_o, ln_active_o} !== 3'b000)
                $fatal(1, "init must work with active=0");
            if ((stream_en !== 1'b0) || (last_beat !== 1'b0))
                $fatal(1, "input launched during SETUP");
            @(negedge clk_i);
            start_i = 1'b0;
            beat_valid_i = 1'b0;
            // All live external fields change. Stored values must remain intact.
            op_i = 3'b111;
            m_i = 10'd0;
            n_i = 10'd0;
            seq_len_i = 7'd0;
            layer_i = ~layer_i;
            num_ln_i = ~num_ln_i;
            req_mult_i = ~req_mult_i;
            req_shamt_i = ~req_shamt_i;
            transpose_i = ~transpose_i;
            write_sel_i = ~write_sel_i;
            write_base_i = ~write_base_i;
            res_base_i = ~res_base_i;
            ln_base_i = ~ln_base_i;
            @(posedge clk_i); #1;
            if ((setup_o !== 1'b0) || (busy_o !== 1'b1) || (done_o !== 1'b0))
                $fatal(1, "bad CORE_RUN entry");
            if ({rq_gelu_active_o, softmax_active_o, ln_active_o} !== group_bits)
                $fatal(1, "wrong active child");
            if ({rq_gelu_init_o, softmax_init_o, ln_init_o} !== 3'b000)
                $fatal(1, "init exceeded one setup cycle");
            if ((command_op == `VFU_RQ) && (rq_gelu_core_op_o !== `VFU_OP_RQ))
                $fatal(1, "wrong RQ core opcode");
            if ((command_op == `VFU_GELU) && (rq_gelu_core_op_o !== `VFU_OP_GELU))
                $fatal(1, "wrong GELU core opcode");
            command_count = command_count + 1;
        end
    endtask

    task automatic check_complete_edge;
        begin
            #1;
            if ((done_o !== 1'b1) || (busy_o !== 1'b1))
                $fatal(1, "completion must be forwarded with busy still high");
            check_descriptor();
            @(posedge clk_i); #1;
            if ((done_o !== 1'b0) || (busy_o !== 1'b0) || (setup_o !== 1'b0))
                $fatal(1, "completion must return directly to idle");
            if ({rq_gelu_active_o, softmax_active_o, ln_active_o} !== 3'b000)
                $fatal(1, "active child remained enabled in idle");
            check_descriptor(); // No idle clearing of the live descriptor.
            commit_done_i = 1'b0;
            softmax_done_i = 1'b0;
            ln_done_i = 1'b0;
            beat_valid_i = 1'b0;
        end
    endtask

    task automatic run_rq_gelu(input [2:0] command_op,
                              input integer tokens,
                              input integer features,
                              input integer original_tokens);
        integer b, before_count;
        begin
            before_count = accepted_beats;
            launch_command(command_op, tokens, features, original_tokens);
            for (b = 0; b < expected_beats; b = b + 1) begin
                if ((b % 7) == 3) begin
                    @(negedge clk_i);
                    beat_valid_i = 1'b0;
                    #1;
                    if (last_beat !== 1'b0) $fatal(1, "bubble marked last");
                    @(posedge clk_i); #1;
                    if (done_o !== 1'b0) $fatal(1, "bubble completed command");
                end
                @(negedge clk_i);
                beat_valid_i = 1'b1;
                #1;
                if (stream_en !== 1'b1) $fatal(1, "stream stopped early");
                if (last_beat !== (b == expected_beats - 1))
                    $fatal(1, "bad accepted-beat last at %0d/%0d", b, expected_beats);
                @(posedge clk_i); #1;
            end
            if ((accepted_beats - before_count) != expected_beats)
                $fatal(1, "input beat count mismatch");
            // Final input only enters DRAIN. Deliberately keep input available.
            repeat (4) begin
                @(negedge clk_i); #1;
                if ((stream_en !== 1'b0) || (last_beat !== 1'b0) ||
                    (done_o !== 1'b0) || (busy_o !== 1'b1) || (rq_gelu_active_o !== 1'b1))
                    $fatal(1, "bad drain/late-commit behavior");
                @(posedge clk_i); #1;
            end
            if ((accepted_beats - before_count) != expected_beats)
                $fatal(1, "extra input accepted during drain");
            @(negedge clk_i);
            commit_done_i = 1'b1;
            check_complete_edge();
        end
    endtask

    task automatic run_stub(input [2:0] command_op, input integer tokens);
        begin
            launch_command(command_op, tokens, 64, tokens - 1);
            repeat (4) begin
                @(posedge clk_i); #1;
                if ((done_o !== 1'b0) || (busy_o !== 1'b1))
                    $fatal(1, "unselected child done finished command");
            end
            @(negedge clk_i);
            if (command_op == `VFU_LN) ln_done_i = 1'b1;
            else softmax_done_i = 1'b1;
            check_complete_edge();
        end
    endtask

    initial begin
        if ($value$plusargs("NEGATIVE=%d", negative_case)) begin end
        start_i = 1'b1; // Reset must never advertise command acceptance.
        repeat (3) @(posedge clk_i);
        #1;
        if (cmd_accept_o !== 1'b0) $fatal(1, "accept advertised during reset");
        if ((busy_o !== 1'b0) || (done_o !== 1'b0) || (setup_o !== 1'b0) ||
            ({rq_gelu_init_o, softmax_init_o, ln_init_o} !== 3'b000) ||
            ({rq_gelu_active_o, softmax_active_o, ln_active_o} !== 3'b000))
            $fatal(1, "reset did not establish inactive idle");
        @(negedge clk_i); start_i = 1'b0; rst_ni = 1'b1;

        if (negative_case == 1) begin
            @(negedge clk_i); op_i = 3'b111; start_i = 1'b1;
            @(posedge clk_i); #1;
            $fatal(1, "negative test: reserved opcode was not rejected");
        end
        if (negative_case == 2) begin
            launch_command(`VFU_SM, 16, 16, 2);
            @(negedge clk_i); op_i = `VFU_RQ; start_i = 1'b1;
            @(posedge clk_i); #1;
            $fatal(1, "negative test: busy start was not rejected");
        end

        run_rq_gelu(`VFU_RQ,   16,   1,  2);
        run_rq_gelu(`VFU_GELU, 32, 128, 17);
        run_rq_gelu(`VFU_RQ,   48,  64, 33);
        run_rq_gelu(`VFU_RQ,   64, 512, 63);
        run_rq_gelu(`VFU_GELU, 64, 512, 64);
        run_stub(`VFU_SM,    64);
        run_stub(`VFU_SM_NL, 64);
        run_stub(`VFU_LN,    16);
        run_rq_gelu(`VFU_GELU, 16, 1, 16);
        repeat (3) @(posedge clk_i);
        #1;
        if ((init_count != command_count) || (completion_count != command_count))
            $fatal(1, "init/done event counts mismatch: %0d/%0d/%0d", init_count, completion_count, command_count);
        if (accepted_beats != 4546) $fatal(1, "total accepted-beat coverage mismatch");
        $display("[PASS] common dispatcher: %0d commands, %0d RQ/GELU beats; config hold, init/active, selected done, delayed commit, 2048-beat limits", command_count, accepted_beats);
        $finish;
    end

    initial begin
        #2000000;
        $fatal(1, "common test timeout");
    end
endmodule
