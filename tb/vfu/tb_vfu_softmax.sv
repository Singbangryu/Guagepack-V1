`timescale 1ns/1ps

module tb_vfu_softmax;
    reg clk_i;
    reg rst_ni = 1'b0;
    reg ce_i = 1'b1;

    reg rowmax_valid_i = 1'b0;
    reg rowmax_clear_i = 1'b0;
    reg rowmax_last_i = 1'b0;
    reg [15:0] score_lane_valid_i = 16'd0;
    reg [383:0] score_i = 384'd0;

    wire score_accept_o;
    wire rowmax_done_o;
    wire [383:0] rowmax_o;
    wire [15:0] rowmax_has_valid_o;

    reg rowsum_valid_i = 1'b0;
    reg rowsum_clear_i = 1'b0;
    reg rowsum_last_i = 1'b0;
    reg [15:0] e_lane_valid_i = 16'd0;
    reg [127:0] e_i = 128'd0;

    wire e_accept_o;
    wire rowsum_done_o;
    wire [207:0] rowsum_o;
    wire [15:0] rowsum_zero_o;

    integer key;
    integer lane;
    integer errors = 0;
    integer score_accept_count;
    integer e_accept_count;
    integer expected_max [0:15];
    integer expected_sum [0:15];
    reg [15:0] expected_has_valid;
    reg [15:0] beat_mask;
    integer score_value;
    integer e_value;

    initial clk_i = 1'b0;
    always #5 clk_i = ~clk_i;

    vfu_softmax dut (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .ce_i(ce_i),

        .rowmax_valid_i(rowmax_valid_i),
        .rowmax_clear_i(rowmax_clear_i),
        .rowmax_last_i(rowmax_last_i),
        .score_lane_valid_i(score_lane_valid_i),
        .score_i(score_i),
        .score_accept_o(score_accept_o),
        .rowmax_done_o(rowmax_done_o),
        .rowmax_o(rowmax_o),
        .rowmax_has_valid_o(rowmax_has_valid_o),

        .rowsum_valid_i(rowsum_valid_i),
        .rowsum_clear_i(rowsum_clear_i),
        .rowsum_last_i(rowsum_last_i),
        .e_lane_valid_i(e_lane_valid_i),
        .e_i(e_i),
        .e_accept_o(e_accept_o),
        .rowsum_done_o(rowsum_done_o),
        .rowsum_o(rowsum_o),
        .rowsum_zero_o(rowsum_zero_o)
    );

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            score_accept_count <= 0;
            e_accept_count <= 0;
        end else begin
            if (score_accept_o)
                score_accept_count <= score_accept_count + 1;
            if (e_accept_o)
                e_accept_count <= e_accept_count + 1;
        end
    end

    task automatic send_score(
        input bit clear,
        input bit last,
        input logic [15:0] mask
    );
        begin
            @(negedge clk_i);
            rowmax_valid_i = 1'b1;
            rowmax_clear_i = clear;
            rowmax_last_i = last;
            score_lane_valid_i = mask;
            @(posedge clk_i);
            #1;
            rowmax_valid_i = 1'b0;
            rowmax_clear_i = 1'b0;
            rowmax_last_i = 1'b0;
            score_lane_valid_i = 16'd0;
        end
    endtask

    task automatic send_e(
        input bit clear,
        input bit last,
        input logic [15:0] mask
    );
        begin
            @(negedge clk_i);
            rowsum_valid_i = 1'b1;
            rowsum_clear_i = clear;
            rowsum_last_i = last;
            e_lane_valid_i = mask;
            @(posedge clk_i);
            #1;
            rowsum_valid_i = 1'b0;
            rowsum_clear_i = 1'b0;
            rowsum_last_i = 1'b0;
            e_lane_valid_i = 16'd0;
        end
    endtask

    task automatic check_rowmax_group;
        begin
            if (!rowmax_done_o) begin
                $display("FAIL ROWMAX done missing on accepted last beat");
                errors = errors + 1;
            end

            for (lane = 0; lane < 16; lane = lane + 1) begin
                if (rowmax_has_valid_o[lane] !== expected_has_valid[lane]) begin
                    $display("FAIL ROWMAX lane=%0d has_valid=%b expected=%b",
                             lane, rowmax_has_valid_o[lane],
                             expected_has_valid[lane]);
                    errors = errors + 1;
                end

                if ($signed({{8{rowmax_o[lane*24 + 23]}},
                              rowmax_o[lane*24 +: 24]})
                    !== expected_max[lane]) begin
                    $display("FAIL ROWMAX lane=%0d got=%0d expected=%0d",
                             lane, $signed(rowmax_o[lane*24 +: 24]),
                             expected_max[lane]);
                    errors = errors + 1;
                end
            end
        end
    endtask

    task automatic check_rowsum_group;
        begin
            if (!rowsum_done_o) begin
                $display("FAIL ROWSUM done missing on accepted last beat");
                errors = errors + 1;
            end

            for (lane = 0; lane < 16; lane = lane + 1) begin
                if ({19'd0, rowsum_o[lane*13 +: 13]}
                    !== expected_sum[lane]) begin
                    $display("FAIL ROWSUM lane=%0d got=%0d expected=%0d",
                             lane, rowsum_o[lane*13 +: 13],
                             expected_sum[lane]);
                    errors = errors + 1;
                end

                if (rowsum_zero_o[lane] !== (expected_sum[lane] == 0)) begin
                    $display("FAIL ROWSUM lane=%0d zero=%b expected=%b",
                             lane, rowsum_zero_o[lane],
                             (expected_sum[lane] == 0));
                    errors = errors + 1;
                end
            end
        end
    endtask

    initial begin
        repeat (3) @(posedge clk_i);
        @(negedge clk_i);
        rst_ni = 1'b1;

        // ---------------------------------------------------------------------
        // ROWMAX: one score-scratch word per key, 16 independent query rows.
        // Keys 48..63 are padding. Query lane 15 is an all-masked query.
        // ---------------------------------------------------------------------
        expected_has_valid = 16'd0;
        for (lane = 0; lane < 16; lane = lane + 1)
            expected_max[lane] = 0;

        for (key = 0; key < 64; key = key + 1) begin
            score_i = 384'd0;
            beat_mask = 16'd0;

            for (lane = 0; lane < 16; lane = lane + 1) begin
                // Different maxima per query lane. Lane 3 has an invalid +max
                // payload at key 50 to prove that padding never participates.
                score_value = (key * (lane + 1)) - (lane * 37) - 500;
                if ((lane == 3) && (key == 50))
                    score_value = 8388607;

                score_i[lane*24 +: 24] =
                    $unsigned(24'(score_value));

                if ((key < 48) && (lane != 15)) begin
                    beat_mask[lane] = 1'b1;
                    if (!expected_has_valid[lane]
                        || (score_value > expected_max[lane])) begin
                        expected_max[lane] = score_value;
                    end
                    expected_has_valid[lane] = 1'b1;
                end
            end

            send_score(key == 0, key == 63, beat_mask);
        end
        check_rowmax_group();

        // Back-to-back single-key groups overwrite all 16 states on clear.
        score_i = 384'd0;
        expected_has_valid = 16'hffff;
        for (lane = 0; lane < 16; lane = lane + 1) begin
            score_value = lane - 100;
            score_i[lane*24 +: 24] = $unsigned(24'(score_value));
            expected_max[lane] = score_value;
        end
        send_score(1'b1, 1'b1, 16'hffff);
        check_rowmax_group();

        score_i = 384'd0;
        expected_has_valid = 16'h7fff;
        for (lane = 0; lane < 16; lane = lane + 1) begin
            score_value = 1000 - lane;
            score_i[lane*24 +: 24] = $unsigned(24'(score_value));
            expected_max[lane] = (lane == 15) ? 0 : score_value;
        end
        send_score(1'b1, 1'b1, 16'h7fff);
        check_rowmax_group();

        // Minimum S24 is a valid value, distinct from an invalid lane.
        score_i = 384'd0;
        score_i[23:0] = 24'h800000;
        expected_has_valid = 16'h0001;
        for (lane = 0; lane < 16; lane = lane + 1)
            expected_max[lane] = 0;
        expected_max[0] = -8388608;
        send_score(1'b1, 1'b1, 16'h0001);
        check_rowmax_group();

        // CE stall: held input is neither stored nor accumulated until accept.
        @(posedge clk_i); // retire previous done
        #1;
        score_i = 384'd0;
        expected_has_valid = 16'hffff;
        for (lane = 0; lane < 16; lane = lane + 1) begin
            score_value = lane + 77;
            score_i[lane*24 +: 24] = $unsigned(24'(score_value));
            expected_max[lane] = score_value;
        end

        @(negedge clk_i);
        ce_i = 1'b0;
        rowmax_valid_i = 1'b1;
        rowmax_clear_i = 1'b1;
        rowmax_last_i = 1'b1;
        score_lane_valid_i = 16'hffff;
        repeat (2) begin
            @(posedge clk_i);
            #1;
            if (score_accept_o || rowmax_done_o) begin
                $display("FAIL ROWMAX accepted/changed during CE stall");
                errors = errors + 1;
            end
        end
        @(negedge clk_i);
        ce_i = 1'b1;
        @(posedge clk_i);
        #1;
        rowmax_valid_i = 1'b0;
        rowmax_clear_i = 1'b0;
        rowmax_last_i = 1'b0;
        score_lane_valid_i = 16'd0;
        check_rowmax_group();

        // ---------------------------------------------------------------------
        // ROWSUM: 64 key beats, 16 independent U13 sums.
        // lane0 reaches the exact worst-case 64*127 = 8128.
        // lane15 remains an all-masked/padding query with L=0.
        // ---------------------------------------------------------------------
        for (lane = 0; lane < 16; lane = lane + 1)
            expected_sum[lane] = 0;

        for (key = 0; key < 64; key = key + 1) begin
            e_i = 128'd0;
            beat_mask = 16'd0;

            for (lane = 0; lane < 16; lane = lane + 1) begin
                if (lane == 0)
                    e_value = 127;
                else if (lane == 1)
                    e_value = key;
                else if (lane == 2)
                    e_value = 100;
                else
                    e_value = (key + lane) & 127;

                e_i[lane*8 +: 8] = {1'b0, e_value[6:0]};

                if ((lane != 15) && ((lane != 2) || (key < 10))) begin
                    beat_mask[lane] = 1'b1;
                    expected_sum[lane] = expected_sum[lane] + e_value;
                end
            end

            send_e(key == 0, key == 63, beat_mask);
        end
        check_rowsum_group();

        if (expected_sum[0] !== 8128 || expected_sum[1] !== 2016
            || expected_sum[2] !== 1000 || expected_sum[15] !== 0) begin
            $display("FAIL testbench ROWSUM reference constants");
            errors = errors + 1;
        end

        if (score_accept_count !== 68) begin
            $display("FAIL score_accept_count=%0d expected=68",
                     score_accept_count);
            errors = errors + 1;
        end

        if (e_accept_count !== 64) begin
            $display("FAIL e_accept_count=%0d expected=64", e_accept_count);
            errors = errors + 1;
        end

        if (errors == 0)
            $display("PASS: vfu_softmax 16-lane rowmax/rowsum tests");
        else
            $display("FAIL: vfu_softmax errors=%0d", errors);

        $finish;
    end

endmodule
