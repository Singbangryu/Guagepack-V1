`timescale 1ns/1ps

module tb_vfu_vcorner_ff;
    localparam integer NUM_GROUPS = 10;
    localparam integer GROUP_BEATS = 16;
    localparam integer TOTAL_BEATS = NUM_GROUPS * GROUP_BEATS;

    reg clk_i = 1'b0;
    reg rst_ni = 1'b0;

    reg         in_valid_i = 1'b0;
    wire        in_ready_o;
    reg  [3:0]  in_col_i = 4'd0;
    reg  [4:0]  in_tag_i = 5'd0;
    reg  [127:0] in_data_i = 128'd0;

    wire         out_valid_o;
    reg          out_ready_i = 1'b1;
    wire [3:0]   out_row_o;
    wire [4:0]   out_tag_o;
    wire [127:0] out_data_o;
    wire         out_last_o;
    wire         idle_o;

    reg [3:0] column_order [0:NUM_GROUPS-1][0:15];
    reg [4:0] group_tag [0:NUM_GROUPS-1];
    reg [7:0] group_data [0:NUM_GROUPS-1][0:15][0:15];
    integer tag_pool [0:31];

    integer errors;
    integer test_seed;
    integer seed;
    integer random_value;
    integer control_seed;
    integer control_random;
    integer group_idx;
    integer beat_idx;
    integer token_idx;
    integer feature_idx;
    integer swap_idx;
    integer swap_value;

    integer cycle_count;
    integer input_accept_count;
    integer output_accept_count;
    integer expected_output_group;
    integer expected_output_row;
    integer last_input_fire_cycle;
    integer simultaneous_cycles;
    integer input_stall_cycles;
    integer output_stall_cycles;
    integer random_output_stall_cycles;

    integer producer_beat_count;
    reg [15:0] producer_seen;
    reg [4:0] producer_tag;

    reg input_hold_pending;
    reg [137:0] input_hold_bundle;
    reg output_hold_pending;
    reg [138:0] output_hold_bundle;

    wire input_fire;
    wire output_fire;
    wire selected_write_full;
    wire selected_read_full;

    assign input_fire = in_valid_i && in_ready_o;
    assign output_fire = out_valid_o && out_ready_i;

    assign selected_write_full = u_dut.write_bank_r
                               ? u_dut.bank_1_full_r
                               : u_dut.bank_0_full_r;
    assign selected_read_full = u_dut.read_bank_r
                              ? u_dut.bank_1_full_r
                              : u_dut.bank_0_full_r;

    always #5 clk_i = ~clk_i;

    vfu_vcorner_ff u_dut (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .in_valid_i(in_valid_i),
        .in_ready_o(in_ready_o),
        .in_col_i(in_col_i),
        .in_tag_i(in_tag_i),
        .in_data_i(in_data_i),
        .out_valid_o(out_valid_o),
        .out_ready_i(out_ready_i),
        .out_row_o(out_row_o),
        .out_tag_o(out_tag_o),
        .out_data_o(out_data_o),
        .out_last_o(out_last_o),
        .idle_o(idle_o)
    );

    task automatic drive_group;
        input integer current_group;
        integer current_beat;
        integer current_token;
        reg [3:0] current_col;
        reg accepted;
        begin
            for (current_beat = 0; current_beat < 16;
                 current_beat = current_beat + 1) begin
                @(negedge clk_i);
                current_col = column_order[current_group][current_beat];
                in_valid_i = 1'b1;
                in_col_i = current_col;
                in_tag_i = group_tag[current_group];
                in_data_i = 128'd0;
                for (current_token = 0; current_token < 16;
                     current_token = current_token + 1) begin
                    in_data_i[(current_token * 8) +: 8]
                        = group_data[current_group][current_token][current_col];
                end

                accepted = 1'b0;
                while (!accepted) begin
                    @(posedge clk_i);
                    accepted = rst_ni && input_fire;
                end
            end
        end
    endtask

    // Keep the first two groups as one uninterrupted 32-beat input burst.
    // Then hold the output long enough to fill both banks and force input
    // backpressure.  Randomized output backpressure follows that directed case.
    always @(negedge clk_i) begin
        if (!rst_ni) begin
            out_ready_i = 1'b1;
        end else if (input_accept_count < 32) begin
            out_ready_i = 1'b1;
        end else if (input_accept_count < 48) begin
            out_ready_i = 1'b0;
        end else begin
            control_random = $random(control_seed);
            out_ready_i = (control_random[1:0] != 2'd0);
        end
    end

    // Input contract checker and first-burst continuity checker.
    always @(posedge clk_i) begin
        if (!rst_ni) begin
            cycle_count = 0;
            input_accept_count = 0;
            last_input_fire_cycle = -1;
            producer_beat_count = 0;
            producer_seen = 16'd0;
            producer_tag = 5'd0;
            input_hold_pending = 1'b0;
            input_hold_bundle = 138'd0;
            input_stall_cycles = 0;
        end else begin
            cycle_count = cycle_count + 1;

            if (input_hold_pending &&
                ({in_valid_i, in_col_i, in_tag_i, in_data_i}
                 !== input_hold_bundle)) begin
                $display("FAIL: input changed before acceptance at cycle %0d",
                         cycle_count);
                errors = errors + 1;
            end

            if (input_fire) begin
                if ((input_accept_count > 0) &&
                    (input_accept_count < 32) &&
                    (cycle_count != (last_input_fire_cycle + 1))) begin
                    $display("FAIL: first 32-beat input burst was interrupted");
                    errors = errors + 1;
                end
                last_input_fire_cycle = cycle_count;
                input_accept_count = input_accept_count + 1;

                if (producer_beat_count == 0) begin
                    producer_seen = 16'd0;
                    producer_tag = in_tag_i;
                end else if (in_tag_i !== producer_tag) begin
                    $display("FAIL: input tag changed within a group");
                    errors = errors + 1;
                end

                if (producer_seen[in_col_i]) begin
                    $display("FAIL: duplicate input column %0d", in_col_i);
                    errors = errors + 1;
                end
                producer_seen[in_col_i] = 1'b1;

                if (producer_beat_count == 15) begin
                    if (producer_seen !== 16'hffff) begin
                        $display("FAIL: input group did not contain 16 distinct columns");
                        errors = errors + 1;
                    end
                    producer_beat_count = 0;
                    producer_seen = 16'd0;
                end else begin
                    producer_beat_count = producer_beat_count + 1;
                end
            end

            if (in_valid_i && !input_fire) begin
                input_stall_cycles = input_stall_cycles + 1;
                input_hold_pending = 1'b1;
                input_hold_bundle = {in_valid_i, in_col_i, in_tag_i, in_data_i};
            end else begin
                input_hold_pending = 1'b0;
            end
        end
    end

    // Output scoreboard, stall stability, and bank-ownership checks.
    always @(posedge clk_i) begin
        if (!rst_ni) begin
            output_accept_count = 0;
            expected_output_group = 0;
            expected_output_row = 0;
            simultaneous_cycles = 0;
            output_stall_cycles = 0;
            random_output_stall_cycles = 0;
            output_hold_pending = 1'b0;
            output_hold_bundle = 139'd0;
        end else begin
            if (in_ready_o !== ~selected_write_full) begin
                $display("FAIL: illegal write-bank ready ownership");
                errors = errors + 1;
            end
            if (out_valid_o !== selected_read_full) begin
                $display("FAIL: illegal read-bank valid ownership");
                errors = errors + 1;
            end
            if (input_fire && selected_write_full) begin
                $display("FAIL: attempted overwrite of a FULL bank");
                errors = errors + 1;
            end
            if ((u_dut.write_col_ptr_r != 0) && selected_write_full) begin
                $display("FAIL: partially filled bank is marked FULL");
                errors = errors + 1;
            end
            if (u_dut.bank_0_full_r && u_dut.bank_1_full_r &&
                (u_dut.write_bank_r != u_dut.read_bank_r)) begin
                $display("FAIL: illegal full-bank pointer ownership");
                errors = errors + 1;
            end

            if (output_hold_pending &&
                ({out_valid_o, out_data_o, out_row_o, out_tag_o, out_last_o}
                 !== output_hold_bundle)) begin
                $display("FAIL: output changed while stalled");
                errors = errors + 1;
            end

            if (output_fire) begin
                if (expected_output_group >= NUM_GROUPS) begin
                    $display("FAIL: unexpected extra output beat");
                    errors = errors + 1;
                end else begin
                    if (out_row_o !== expected_output_row[3:0]) begin
                        $display("FAIL: row order group=%0d expected=%0d got=%0d",
                                 expected_output_group, expected_output_row,
                                 out_row_o);
                        errors = errors + 1;
                    end
                    if (out_tag_o !== group_tag[expected_output_group]) begin
                        $display("FAIL: tag mismatch group=%0d row=%0d expected=%0d got=%0d",
                                 expected_output_group, expected_output_row,
                                 group_tag[expected_output_group], out_tag_o);
                        errors = errors + 1;
                    end
                    if (out_last_o !== (expected_output_row == 15)) begin
                        $display("FAIL: last mismatch group=%0d row=%0d",
                                 expected_output_group, expected_output_row);
                        errors = errors + 1;
                    end
                    for (feature_idx = 0; feature_idx < 16;
                         feature_idx = feature_idx + 1) begin
                        if (out_data_o[(feature_idx * 8) +: 8]
                            !== group_data[expected_output_group]
                                         [expected_output_row][feature_idx]) begin
                            $display("FAIL: data mismatch group=%0d row=%0d feature=%0d expected=%02x got=%02x",
                                     expected_output_group, expected_output_row,
                                     feature_idx,
                                     group_data[expected_output_group]
                                               [expected_output_row][feature_idx],
                                     out_data_o[(feature_idx * 8) +: 8]);
                            errors = errors + 1;
                        end
                    end
                end

                output_accept_count = output_accept_count + 1;
                if (expected_output_row == 15) begin
                    expected_output_row = 0;
                    expected_output_group = expected_output_group + 1;
                end else begin
                    expected_output_row = expected_output_row + 1;
                end
            end

            if (input_fire && output_fire) begin
                simultaneous_cycles = simultaneous_cycles + 1;
                if (u_dut.write_bank_r == u_dut.read_bank_r) begin
                    $display("FAIL: simultaneous fill/drain used the same bank");
                    errors = errors + 1;
                end
            end

            if (out_valid_o && !output_fire) begin
                output_stall_cycles = output_stall_cycles + 1;
                if (input_accept_count >= 48)
                    random_output_stall_cycles = random_output_stall_cycles + 1;
                output_hold_pending = 1'b1;
                output_hold_bundle = {out_valid_o, out_data_o, out_row_o,
                                      out_tag_o, out_last_o};
            end else begin
                output_hold_pending = 1'b0;
            end
        end
    end

    initial begin
        errors = 0;
        test_seed = 32'h4f27a19d;
        random_value = $value$plusargs("SEED=%d", test_seed);
        seed = test_seed;
        control_seed = test_seed ^ 32'h27d4eb2d;
        random_value = $urandom(seed);

        for (group_idx = 0; group_idx < 32; group_idx = group_idx + 1)
            tag_pool[group_idx] = group_idx;

        for (group_idx = 31; group_idx > 0; group_idx = group_idx - 1) begin
            swap_idx = $urandom_range(group_idx, 0);
            swap_value = tag_pool[group_idx];
            tag_pool[group_idx] = tag_pool[swap_idx];
            tag_pool[swap_idx] = swap_value;
        end

        for (group_idx = 0; group_idx < NUM_GROUPS;
             group_idx = group_idx + 1) begin
            group_tag[group_idx] = tag_pool[group_idx];

            for (beat_idx = 0; beat_idx < 16; beat_idx = beat_idx + 1)
                column_order[group_idx][beat_idx] = beat_idx[3:0];

            if (group_idx == 1) begin
                for (beat_idx = 0; beat_idx < 16; beat_idx = beat_idx + 1)
                    column_order[group_idx][beat_idx] = 15 - beat_idx;
            end else if (group_idx >= 2) begin
                for (beat_idx = 15; beat_idx > 0; beat_idx = beat_idx - 1) begin
                    swap_idx = $urandom_range(beat_idx, 0);
                    swap_value = column_order[group_idx][beat_idx];
                    column_order[group_idx][beat_idx]
                        = column_order[group_idx][swap_idx];
                    column_order[group_idx][swap_idx] = swap_value[3:0];
                end
            end

            for (token_idx = 0; token_idx < 16; token_idx = token_idx + 1) begin
                for (feature_idx = 0; feature_idx < 16;
                     feature_idx = feature_idx + 1) begin
                    group_data[group_idx][token_idx][feature_idx]
                        = ((group_idx * 53) + (token_idx * 17)
                           + (feature_idx * 29) + (token_idx * feature_idx));
                end
            end
        end

        repeat (4) @(posedge clk_i);
        @(negedge clk_i);
        rst_ni = 1'b1;

        for (group_idx = 0; group_idx < NUM_GROUPS;
             group_idx = group_idx + 1)
            drive_group(group_idx);

        @(negedge clk_i);
        in_valid_i = 1'b0;
        in_col_i = 4'd0;
        in_tag_i = 5'd0;
        in_data_i = 128'd0;

        wait (expected_output_group == NUM_GROUPS);
        repeat (3) @(posedge clk_i);
        #1;

        if (input_accept_count != TOTAL_BEATS) begin
            $display("FAIL: input beat count expected=%0d got=%0d",
                     TOTAL_BEATS, input_accept_count);
            errors = errors + 1;
        end
        if (output_accept_count != TOTAL_BEATS) begin
            $display("FAIL: output beat count expected=%0d got=%0d",
                     TOTAL_BEATS, output_accept_count);
            errors = errors + 1;
        end
        if (producer_beat_count != 0) begin
            $display("FAIL: input producer checker ended mid-group");
            errors = errors + 1;
        end
        if (simultaneous_cycles == 0) begin
            $display("FAIL: no simultaneous different-bank fill/drain observed");
            errors = errors + 1;
        end
        if (input_stall_cycles == 0) begin
            $display("FAIL: directed input backpressure was not observed");
            errors = errors + 1;
        end
        if (output_stall_cycles == 0) begin
            $display("FAIL: output backpressure was not observed");
            errors = errors + 1;
        end
        if (random_output_stall_cycles == 0) begin
            $display("FAIL: randomized output backpressure was not observed");
            errors = errors + 1;
        end
        if (!idle_o) begin
            $display("FAIL: idle_o did not assert after all groups drained");
            errors = errors + 1;
        end

        if (errors == 0) begin
            $display("PASS: vfu_vcorner_ff seed=%0d groups=%0d beats=%0d simultaneous=%0d input_stalls=%0d output_stalls=%0d random_output_stalls=%0d",
                     test_seed, NUM_GROUPS, TOTAL_BEATS, simultaneous_cycles,
                     input_stall_cycles, output_stall_cycles,
                     random_output_stall_cycles);
            $finish;
        end else begin
            $fatal(1, "FAIL: vfu_vcorner_ff errors=%0d", errors);
        end
    end

    initial begin
        repeat (20000) @(posedge clk_i);
        $fatal(1, "FAIL: vfu_vcorner_ff timeout");
    end

endmodule
