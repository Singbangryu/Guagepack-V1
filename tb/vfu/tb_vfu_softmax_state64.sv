`timescale 1ns/1ps

module tb_vfu_softmax_state64;
    reg          clk_i = 1'b0;
    reg          rst_ni = 1'b0;
    reg          ce_i = 1'b0;
    reg          clear_i = 1'b0;
    reg          wr_en_i = 1'b0;
    reg  [1:0]   wr_tile_i = 2'd0;
    reg          wr_type_i = 1'b0;
    reg  [383:0] wr_data_i = 384'd0;
    reg  [1:0]   rd_tile_i = 2'd0;

    wire         rd_valid_o;
    wire         rd_type_o;
    wire [383:0] rd_data_o;

    reg [383:0] model_data [0:3];
    reg [3:0]   model_valid;
    reg [3:0]   model_type;

    integer seed;
    integer rng_seed;
    integer seed_junk;
    integer errors = 0;
    integer checks = 0;
    integer tile;
    integer lane;
    integer cycle;
    reg [31:0] random_word;
    reg [383:0] directed_old;
    reg [383:0] directed_new;

    always #5 clk_i = ~clk_i;

    vfu_softmax_state64 dut (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .ce_i(ce_i),
        .clear_i(clear_i),
        .wr_en_i(wr_en_i),
        .wr_tile_i(wr_tile_i),
        .wr_type_i(wr_type_i),
        .wr_data_i(wr_data_i),
        .rd_tile_i(rd_tile_i),
        .rd_valid_o(rd_valid_o),
        .rd_type_o(rd_type_o),
        .rd_data_o(rd_data_o)
    );

    function automatic [383:0] make_pattern;
        input integer tile_index;
        input integer salt;
        integer pattern_lane;
        reg [23:0] lane_word;
        begin
            make_pattern = 384'd0;
            for (pattern_lane = 0; pattern_lane < 16;
                 pattern_lane = pattern_lane + 1) begin
                lane_word = salt * 97 + tile_index * 31
                          + pattern_lane * 7;
                make_pattern[pattern_lane*24 +: 24] = lane_word;
            end
        end
    endfunction

    task automatic check_read;
        input integer tile_index;
        input [255:0] label_text;
        reg expected_valid;
        reg expected_type;
        reg [383:0] expected_data;
        begin
            rd_tile_i = tile_index[1:0];
            #1;
            expected_valid = model_valid[tile_index];
            expected_type  = expected_valid ? model_type[tile_index] : 1'b0;
            expected_data  = expected_valid ? model_data[tile_index] : 384'd0;
            checks = checks + 1;

            if ((rd_valid_o !== expected_valid)
             || (rd_type_o  !== expected_type)
             || (rd_data_o  !== expected_data)) begin
                $display("[FAIL] %0s tile=%0d valid=%b/%b type=%b/%b",
                         label_text, tile_index,
                         rd_valid_o, expected_valid,
                         rd_type_o, expected_type);
                errors = errors + 1;
            end
        end
    endtask

    task automatic check_all_tiles;
        input [255:0] label_text;
        integer check_tile;
        begin
            for (check_tile = 0; check_tile < 4;
                 check_tile = check_tile + 1)
                check_read(check_tile, label_text);
        end
    endtask

    task automatic model_edge;
        begin
            if (!rst_ni) begin
                model_valid = 4'b0000;
                model_type  = 4'b0000;
            end else if (ce_i) begin
                if (clear_i) begin
                    model_valid = 4'b0000;
                    model_type  = 4'b0000;
                end else if (wr_en_i) begin
                    model_data[wr_tile_i] = wr_data_i;
                    model_valid[wr_tile_i] = 1'b1;
                    model_type[wr_tile_i] = wr_type_i;
                end
            end
        end
    endtask

    initial begin
        if (!$value$plusargs("SEED=%d", seed))
            seed = 1;
        rng_seed = seed;
        seed_junk = $urandom(rng_seed);

        model_valid = 4'b0000;
        model_type  = 4'b0000;
        for (tile = 0; tile < 4; tile = tile + 1)
            model_data[tile] = 384'd0;

        // Synchronous reset dominates CE hold and simultaneous clear/write.
        rst_ni = 1'b0;
        ce_i = 1'b0;
        clear_i = 1'b1;
        wr_en_i = 1'b1;
        wr_tile_i = 2'd3;
        wr_type_i = 1'b1;
        wr_data_i = make_pattern(3, 1);
        repeat (2) begin
            @(posedge clk_i);
            #1;
            model_edge();
        end
        check_all_tiles("reset masks data");

        @(negedge clk_i);
        rst_ni = 1'b1;
        ce_i = 1'b1;
        clear_i = 1'b0;

        // Populate every tile, alternating MAX/R metadata.
        for (tile = 0; tile < 4; tile = tile + 1) begin
            wr_en_i = 1'b1;
            wr_tile_i = tile[1:0];
            wr_type_i = tile[0];
            wr_data_i = make_pattern(tile, 10 + tile);
            check_read(tile, "old state before write edge");
            @(posedge clk_i);
            #1;
            model_edge();
            check_read(tile, "new state after write edge");
            @(negedge clk_i);
        end

        wr_en_i = 1'b0;
        check_all_tiles("combinational tile switch");

        // CE=0 holds metadata/data and ignores both clear and write.
        ce_i = 1'b0;
        clear_i = 1'b1;
        wr_en_i = 1'b1;
        wr_tile_i = 2'd2;
        wr_type_i = 1'b0;
        wr_data_i = make_pattern(2, 50);
        @(posedge clk_i);
        #1;
        model_edge();
        check_all_tiles("CE hold");

        // With CE enabled, clear dominates a simultaneous write.
        @(negedge clk_i);
        ce_i = 1'b1;
        clear_i = 1'b1;
        wr_en_i = 1'b1;
        wr_tile_i = 2'd1;
        wr_type_i = 1'b1;
        wr_data_i = make_pattern(1, 60);
        @(posedge clk_i);
        #1;
        model_edge();
        check_all_tiles("clear dominates write");

        // Directed MAX-to-R replacement and same-tile no-bypass boundary.
        @(negedge clk_i);
        clear_i = 1'b0;
        wr_en_i = 1'b1;
        wr_tile_i = 2'd2;
        wr_type_i = 1'b0;
        directed_old = make_pattern(2, 70);
        wr_data_i = directed_old;
        @(posedge clk_i);
        #1;
        model_edge();
        check_read(2, "MAX state");

        @(negedge clk_i);
        wr_type_i = 1'b1;
        directed_new = make_pattern(2, 80);
        for (lane = 0; lane < 16; lane = lane + 1)
            directed_new[lane*24 + 18 +: 6] = 6'd0;
        wr_data_i = directed_new;
        check_read(2, "collision exposes old MAX");
        @(posedge clk_i);
        #1;
        model_edge();
        check_read(2, "following cycle exposes R");

        // Reset again with CE low to prove reset remains highest priority.
        @(negedge clk_i);
        rst_ni = 1'b0;
        ce_i = 1'b0;
        clear_i = 1'b0;
        wr_en_i = 1'b1;
        wr_tile_i = 2'd2;
        @(posedge clk_i);
        #1;
        model_edge();
        check_all_tiles("reset over CE hold");

        @(negedge clk_i);
        rst_ni = 1'b1;
        ce_i = 1'b1;
        wr_en_i = 1'b0;

        // Seeded cycle-by-cycle reference-model stress.  Each cycle checks the
        // combinational pre-edge value and the post-edge priority result.
        for (cycle = 0; cycle < 1000; cycle = cycle + 1) begin
            rst_ni = ($urandom_range(96, 0) != 0);
            ce_i = $urandom_range(3, 0) != 0;
            clear_i = ($urandom_range(19, 0) == 0);
            wr_en_i = ($urandom_range(2, 0) != 0);
            wr_tile_i = $urandom_range(3, 0);
            wr_type_i = $urandom_range(1, 0);
            wr_data_i = 384'd0;
            for (lane = 0; lane < 16; lane = lane + 1) begin
                random_word = $urandom;
                if (wr_type_i)
                    wr_data_i[lane*24 +: 24] = {6'd0, random_word[17:0]};
                else
                    wr_data_i[lane*24 +: 24] = random_word[23:0];
            end
            rd_tile_i = $urandom_range(3, 0);

            check_read(rd_tile_i, "stress pre-edge");
            @(posedge clk_i);
            #1;
            model_edge();
            check_read(rd_tile_i, "stress post-edge");
            @(negedge clk_i);
        end

        if (errors != 0)
            $fatal(1, "[FAIL] seed=%0d checks=%0d errors=%0d",
                   seed, checks, errors);

        $display("[PASS] vfu_softmax_state64 seed=%0d checks=%0d",
                 seed, checks);
        $finish;
    end

endmodule
