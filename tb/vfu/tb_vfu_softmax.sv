`timescale 1ns/1ps

// GaugePack Softmax reduction tests for the key-only masking contract.
//
// One input beat is one key column across all 16 query rows.  key_valid_i is
// therefore a scalar shared by every row.  QEXP is the only owner of key
// masking, so rowsum receives final E7 bytes and simply accumulates all lanes.
// A legal command has at least one valid key; L=0 is not tested as a legal
// result.
module tb_vfu_softmax;
    reg clk_i;
    reg rst_ni = 1'b0;
    reg ce_i = 1'b1;

    reg rowmax_valid_i = 1'b0;
    reg rowmax_clear_i = 1'b0;
    reg rowmax_last_i = 1'b0;
    reg key_valid_i = 1'b0;
    reg [383:0] score_i = 384'd0;

    wire rowmax_done_o;
    wire [383:0] rowmax_o;

    reg rowsum_valid_i = 1'b0;
    reg rowsum_clear_i = 1'b0;
    reg rowsum_last_i = 1'b0;
    reg [127:0] e_i = 128'd0;

    wire rowsum_done_o;
    wire [207:0] rowsum_o;

    integer key;
    integer lane;
    integer errors = 0;
    integer score_value;
    integer e_value;
    integer expected_max [0:15];
    integer expected_sum [0:15];
    reg [383:0] held_rowmax;
    reg [207:0] held_rowsum;

    initial clk_i = 1'b0;
    always #5 clk_i = ~clk_i;

    vfu_softmax dut (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .ce_i(ce_i),

        .rowmax_valid_i(rowmax_valid_i),
        .rowmax_clear_i(rowmax_clear_i),
        .rowmax_last_i(rowmax_last_i),
        .key_valid_i(key_valid_i),
        .score_i(score_i),
        .rowmax_done_o(rowmax_done_o),
        .rowmax_o(rowmax_o),

        .rowsum_valid_i(rowsum_valid_i),
        .rowsum_clear_i(rowsum_clear_i),
        .rowsum_last_i(rowsum_last_i),
        .e_i(e_i),
        .rowsum_done_o(rowsum_done_o),
        .rowsum_o(rowsum_o)
    );

    task automatic send_score(
        input bit clear,
        input bit last,
        input bit key_valid
    );
        begin
            @(negedge clk_i);
            rowmax_valid_i = 1'b1;
            rowmax_clear_i = clear;
            rowmax_last_i = last;
            key_valid_i = key_valid;
            @(posedge clk_i);
            #1;
            rowmax_valid_i = 1'b0;
            rowmax_clear_i = 1'b0;
            rowmax_last_i = 1'b0;
            key_valid_i = 1'b0;
        end
    endtask

    task automatic send_e(
        input bit clear,
        input bit last
    );
        begin
            @(negedge clk_i);
            rowsum_valid_i = 1'b1;
            rowsum_clear_i = clear;
            rowsum_last_i = last;
            @(posedge clk_i);
            #1;
            rowsum_valid_i = 1'b0;
            rowsum_clear_i = 1'b0;
            rowsum_last_i = 1'b0;
        end
    endtask

    task automatic check_rowmax;
        begin
            if (!rowmax_done_o) begin
                $display("FAIL ROWMAX done missing on accepted last beat");
                errors = errors + 1;
            end

            for (lane = 0; lane < 16; lane = lane + 1) begin
                if ($signed(rowmax_o[lane*24 +: 24])
                    != expected_max[lane]) begin
                    $display("FAIL ROWMAX lane=%0d got=%0d expected=%0d",
                             lane, $signed(rowmax_o[lane*24 +: 24]),
                             expected_max[lane]);
                    errors = errors + 1;
                end
            end
        end
    endtask

    task automatic check_rowsum;
        begin
            if (!rowsum_done_o) begin
                $display("FAIL ROWSUM done missing on accepted last beat");
                errors = errors + 1;
            end

            for (lane = 0; lane < 16; lane = lane + 1) begin
                if (rowsum_o[lane*13 +: 13] != expected_sum[lane]) begin
                    $display("FAIL ROWSUM lane=%0d got=%0d expected=%0d",
                             lane, rowsum_o[lane*13 +: 13],
                             expected_sum[lane]);
                    errors = errors + 1;
                end
                if ((expected_sum[lane] < 127)
                    || (expected_sum[lane] > 8128)) begin
                    $display("FAIL testbench generated illegal L lane=%0d L=%0d",
                             lane, expected_sum[lane]);
                    errors = errors + 1;
                end
            end
        end
    endtask

    initial begin
        repeat (3) @(posedge clk_i);
        @(negedge clk_i);
        rst_ni = 1'b1;

        // ------------------------------------------------------------------
        // ROWMAX: keys 48..63 are padding, shared by all query rows.  The
        // invalid key 50 contains a huge score and must affect no row.
        // ------------------------------------------------------------------
        for (lane = 0; lane < 16; lane = lane + 1)
            expected_max[lane] = -8388608;

        for (key = 0; key < 64; key = key + 1) begin
            score_i = 384'd0;
            for (lane = 0; lane < 16; lane = lane + 1) begin
                score_value = key * (lane + 1) - lane * 37 - 500;
                if (key == 50)
                    score_value = 8388607 - lane;
                score_i[lane*24 +: 24] = score_value[23:0];

                if ((key < 48) && (score_value > expected_max[lane]))
                    expected_max[lane] = score_value;
            end
            send_score(key == 0, key == 63, key < 48);
        end
        check_rowmax();

        // done is one accepted-cycle pulse.
        @(posedge clk_i);
        #1;
        if (rowmax_done_o) begin
            $display("FAIL ROWMAX done did not retire");
            errors = errors + 1;
        end

        // S24_MIN is a legal maximum.  No has-valid sideband is required.
        score_i = {16{24'h800000}};
        for (lane = 0; lane < 16; lane = lane + 1)
            expected_max[lane] = -8388608;
        send_score(1'b1, 1'b1, 1'b1);
        check_rowmax();

        // A legal mask need not start at key 0.  An invalid clear beat leaves
        // the S24_MIN identity, and the first later valid key defines rowmax.
        @(posedge clk_i);
        #1;
        score_i = {16{24'h7fffff}}; // ignored payload
        send_score(1'b1, 1'b0, 1'b0);

        score_i = 384'd0;
        for (lane = 0; lane < 16; lane = lane + 1) begin
            score_value = lane - 100;
            score_i[lane*24 +: 24] = score_value[23:0];
            expected_max[lane] = score_value;
        end
        send_score(1'b0, 1'b1, 1'b1);
        check_rowmax();

        // CE stall: an asserted valid/last beat must not be accepted until CE.
        @(posedge clk_i);
        #1;
        score_i = 384'd0;
        for (lane = 0; lane < 16; lane = lane + 1) begin
            score_value = lane + 77;
            score_i[lane*24 +: 24] = score_value[23:0];
            expected_max[lane] = score_value;
        end
        held_rowmax = rowmax_o;

        @(negedge clk_i);
        ce_i = 1'b0;
        rowmax_valid_i = 1'b1;
        rowmax_clear_i = 1'b1;
        rowmax_last_i = 1'b1;
        key_valid_i = 1'b1;
        repeat (2) begin
            @(posedge clk_i);
            #1;
            if ((rowmax_o !== held_rowmax) || rowmax_done_o) begin
                $display("FAIL ROWMAX state changed during CE stall");
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
        key_valid_i = 1'b0;
        check_rowmax();

        // ------------------------------------------------------------------
        // ROWSUM 1: one valid key.  QEXP must emit E=127 at rowmax and zero
        // on all 63 masked keys, so every legal query row has exactly L=127.
        // ------------------------------------------------------------------
        for (lane = 0; lane < 16; lane = lane + 1)
            expected_sum[lane] = 127;

        for (key = 0; key < 64; key = key + 1) begin
            e_i = 128'd0;
            if (key == 23)
                e_i = {16{8'd127}};
            send_e(key == 0, key == 63);
        end
        check_rowsum();

        // ------------------------------------------------------------------
        // ROWSUM 2: mixed shared key mask.  Masking already happened in QEXP;
        // this block sees zero E bytes for invalid key beats.  Key 0 is the
        // rowmax position and supplies E=127 to every row.
        // ------------------------------------------------------------------
        for (lane = 0; lane < 16; lane = lane + 1)
            expected_sum[lane] = 0;

        for (key = 0; key < 64; key = key + 1) begin
            e_i = 128'd0;
            if ((key < 48) && ((key % 5) != 1)) begin
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    if (key == 0)
                        e_value = 127;
                    else
                        e_value = (key + lane) & 7'h7f;
                    e_i[lane*8 +: 8] = {1'b0, e_value[6:0]};
                    expected_sum[lane] = expected_sum[lane] + e_value;
                end
            end

            // Hold one ordinary E beat across a two-cycle CE stall.
            if (key == 17) begin
                held_rowsum = rowsum_o;
                @(negedge clk_i);
                ce_i = 1'b0;
                rowsum_valid_i = 1'b1;
                rowsum_clear_i = 1'b0;
                rowsum_last_i = 1'b0;
                repeat (2) begin
                    @(posedge clk_i);
                    #1;
                    if ((rowsum_o !== held_rowsum) || rowsum_done_o) begin
                        $display("FAIL ROWSUM state changed during CE stall");
                        errors = errors + 1;
                    end
                end
                @(negedge clk_i);
                ce_i = 1'b1;
                @(posedge clk_i);
                #1;
                rowsum_valid_i = 1'b0;
            end else begin
                send_e(key == 0, key == 63);
            end
        end
        check_rowsum();

        // ------------------------------------------------------------------
        // ROWSUM 3: exact U13 upper bound, 64 * E7_MAX = 8128.
        // ------------------------------------------------------------------
        for (lane = 0; lane < 16; lane = lane + 1)
            expected_sum[lane] = 8128;

        for (key = 0; key < 64; key = key + 1) begin
            e_i = {16{8'd127}};
            send_e(key == 0, key == 63);
        end
        check_rowsum();

        if (errors == 0)
            $display("PASS: vfu_softmax key-only mask contract");
        else
            $display("FAIL: vfu_softmax errors=%0d", errors);

        $finish;
    end

endmodule
