`timescale 1ns/1ps

module tb_vfu_row_units;
    reg clk_i = 1'b0;
    reg rst_ni = 1'b0;
    reg ce_i = 1'b1;

    reg valid_i = 1'b0;
    reg clear_i = 1'b0;
    reg last_i = 1'b0;
    reg key_valid_i = 1'b0;
    reg [383:0] score_i = 384'd0;
    reg [127:0] e_i = 128'd0;

    wire rowmax_valid;
    wire exp_commit_valid;
    wire [383:0] rowmax;
    wire [127:0] e_commit;
    wire [207:0] rowsum;

    integer key;
    integer lane;
    integer errors = 0;
    integer score_value;
    integer e_value;

    always #5 clk_i = ~clk_i;

    vfu_rowmax16 u_rowmax (
        .clk_i(clk_i), .rst_ni(rst_ni), .ce_i(ce_i),
        .valid_i(valid_i), .clear_i(clear_i), .last_i(last_i),
        .key_valid_i(key_valid_i), .score_i(score_i),
        .result_valid_o(rowmax_valid), .rowmax_o(rowmax)
    );

    vfu_exp_commit16 u_exp_commit (
        .clk_i(clk_i), .rst_ni(rst_ni), .ce_i(ce_i),
        .valid_i(valid_i), .clear_i(clear_i), .last_i(last_i),
        .key_valid_i(key_valid_i), .e_i(e_i),
        .result_valid_o(exp_commit_valid), .e_o(e_commit),
        .rowsum_o(rowsum)
    );

    initial begin
        repeat (2) @(posedge clk_i);
        @(negedge clk_i);
        rst_ni = 1'b1;

        // PMPU drain order 15 -> 0, seq_len = 5.
        // Keys 15..5 are invalid even though their payloads are large.
        for (key = 15; key >= 0; key = key - 1) begin
            score_i = 384'd0;
            e_i = 128'd0;
            for (lane = 0; lane < 16; lane = lane + 1) begin
                score_value = (key < 5) ? (key * 10 - lane - 100)
                                        : (100000 + key + lane);
                e_value = (key < 5) ? (key + lane + 1) : 127;
                score_i[lane*24 +: 24] = score_value[23:0];
                e_i[lane*8 +: 8] = {1'b0, e_value[6:0]};
            end

            valid_i = 1'b1;
            clear_i = (key == 15);
            last_i = (key == 0);
            key_valid_i = (key < 5);
            @(posedge clk_i);
            #1;

            if (key_valid_i) begin
                if (e_commit !== e_i) begin
                    $display("FAIL: valid E commit mismatch at key=%0d", key);
                    errors = errors + 1;
                end
            end else if (e_commit !== 128'd0) begin
                $display("FAIL: invalid key committed nonzero E at key=%0d", key);
                errors = errors + 1;
            end

            valid_i = 1'b0;
            clear_i = 1'b0;
            last_i = 1'b0;
            key_valid_i = 1'b0;
            @(negedge clk_i);
        end

        if (!rowmax_valid || !exp_commit_valid) begin
            $display("FAIL: result_valid missing");
            errors = errors + 1;
        end

        for (lane = 0; lane < 16; lane = lane + 1) begin
            if ($signed(rowmax[lane*24 +: 24]) != (40 - lane - 100)) begin
                $display("FAIL rowmax lane=%0d got=%0d", lane,
                         $signed(rowmax[lane*24 +: 24]));
                errors = errors + 1;
            end
            if (rowsum[lane*13 +: 13] != (5 * lane + 15)) begin
                $display("FAIL rowsum lane=%0d got=%0d", lane,
                         rowsum[lane*13 +: 13]);
                errors = errors + 1;
            end
        end

        if (errors == 0)
            $display("PASS: vfu rowmax/QEXP commit units");
        else
            $display("FAIL: errors=%0d", errors);
        $finish;
    end
endmodule
