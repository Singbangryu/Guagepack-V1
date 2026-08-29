`timescale 1ns/1ps

module tb_vfu_spad_mem16 #(
    parameter integer RD_LAT = 2
);
    localparam integer DEPTH  = 512;
    localparam integer ADDR_W = 9;

    reg                   clk_i;
    reg                   rst_ni;
    reg                   ce_i;
    reg                   wr_en_i;
    reg  [ADDR_W-1:0]     wr_addr_i;
    reg  [511:0]          wr_data_i;
    reg                   rd_en_i;
    reg  [ADDR_W-1:0]     rd_addr_i;
    wire                  rd_valid_o;
    wire [511:0]          rd_data_o;

    reg [511:0] model_mem [0:DEPTH-1];
    reg         model_known [0:DEPTH-1];

    reg         exp_stage_valid;
    reg [511:0] exp_stage_data;
    reg         exp_stage_data_known;
    reg         exp_out_valid;
    reg [511:0] exp_out_data;
    reg         exp_out_data_known;

    integer seed;
    integer rng_seed;
    integer seed_junk;
    integer checks;
    integer accepted_reads;
    integer accepted_writes;
    integer concurrent_ops;
    integer ce_stalls;
    integer collision_adjustments;
    integer idx;

    reg [511:0] word_zero;
    reg [511:0] word_last;
    reg [511:0] word_reset;
    reg [511:0] word_tmp;
    reg [ADDR_W-1:0] rand_wr_addr;
    reg [ADDR_W-1:0] rand_rd_addr;
    reg rand_ce;
    reg rand_wr;
    reg rand_rd;

    always #5 clk_i = ~clk_i;

    vfu_spad_mem16 #(
        .DEPTH(DEPTH),
        .ADDR_W(ADDR_W),
        .RD_LAT(RD_LAT)
    ) dut (
        .clk_i(clk_i),
        .rst_ni(rst_ni),
        .ce_i(ce_i),
        .wr_en_i(wr_en_i),
        .wr_addr_i(wr_addr_i),
        .wr_data_i(wr_data_i),
        .rd_en_i(rd_en_i),
        .rd_addr_i(rd_addr_i),
        .rd_valid_o(rd_valid_o),
        .rd_data_o(rd_data_o)
    );

    function automatic [511:0] patterned_word;
        input [31:0] base;
        integer lane;
        begin
            patterned_word = 512'd0;
            for (lane = 0; lane < 16; lane = lane + 1)
                patterned_word[lane*32 +: 32] = base ^ (32'h1f123bb5 * lane);
        end
    endfunction

    function automatic [511:0] random_word;
        integer lane;
        begin
            random_word = 512'd0;
            for (lane = 0; lane < 16; lane = lane + 1)
                random_word[lane*32 +: 32] = $urandom;
        end
    endfunction

    task automatic run_cycle;
        input                   next_rst_ni;
        input                   next_ce_i;
        input                   next_wr_en_i;
        input [ADDR_W-1:0]      next_wr_addr_i;
        input [511:0]           next_wr_data_i;
        input                   next_rd_en_i;
        input [ADDR_W-1:0]      next_rd_addr_i;

        reg                     next_stage_valid;
        reg [511:0]             next_stage_data;
        reg                     next_stage_data_known;
        reg                     next_out_valid;
        reg [511:0]             next_out_data;
        reg                     next_out_data_known;
        reg                     expected_valid;
        reg [511:0]             expected_data;
        reg                     expected_data_known;
        begin
            @(negedge clk_i);
            rst_ni    = next_rst_ni;
            ce_i      = next_ce_i;
            wr_en_i   = next_wr_en_i;
            wr_addr_i = next_wr_addr_i;
            wr_data_i = next_wr_data_i;
            rd_en_i   = next_rd_en_i;
            rd_addr_i = next_rd_addr_i;

            next_stage_valid      = exp_stage_valid;
            next_stage_data       = exp_stage_data;
            next_stage_data_known = exp_stage_data_known;
            next_out_valid        = exp_out_valid;
            next_out_data         = exp_out_data;
            next_out_data_known   = exp_out_data_known;

            if (!next_rst_ni) begin
                next_stage_valid = 1'b0;
                next_out_valid   = 1'b0;
            end else if (next_ce_i) begin
                if (next_wr_en_i && next_rd_en_i &&
                    (next_wr_addr_i == next_rd_addr_i)) begin
                    $fatal(1, "illegal same-address collision driven at addr=%0d",
                           next_wr_addr_i);
                end

                next_stage_valid = next_rd_en_i;
                if (next_rd_en_i) begin
                    if (!model_known[next_rd_addr_i])
                        $fatal(1, "read of unwritten address %0d",
                               next_rd_addr_i);
                    next_stage_data = model_mem[next_rd_addr_i];
                    next_stage_data_known = 1'b1;
                    accepted_reads = accepted_reads + 1;
                end

                if (RD_LAT == 2) begin
                    next_out_valid = exp_stage_valid;
                    if (exp_stage_valid) begin
                        next_out_data = exp_stage_data;
                        next_out_data_known = exp_stage_data_known;
                    end
                end

                if (next_wr_en_i) begin
                    model_mem[next_wr_addr_i] = next_wr_data_i;
                    model_known[next_wr_addr_i] = 1'b1;
                    accepted_writes = accepted_writes + 1;
                end

                if (next_wr_en_i && next_rd_en_i)
                    concurrent_ops = concurrent_ops + 1;
            end else begin
                ce_stalls = ce_stalls + 1;
            end

            @(posedge clk_i);
            #1;

            if (RD_LAT == 1) begin
                expected_valid      = next_stage_valid;
                expected_data       = next_stage_data;
                expected_data_known = next_stage_data_known;
            end else begin
                expected_valid      = next_out_valid;
                expected_data       = next_out_data;
                expected_data_known = next_out_data_known;
            end

            if (rd_valid_o !== expected_valid)
                $fatal(1,
                       "rd_valid mismatch RD_LAT=%0d got=%b expected=%b",
                       RD_LAT, rd_valid_o, expected_valid);

            if (expected_data_known && (rd_data_o !== expected_data))
                $fatal(1,
                       "rd_data mismatch RD_LAT=%0d got[63:0]=%h expected[63:0]=%h",
                       RD_LAT, rd_data_o[63:0], expected_data[63:0]);

            exp_stage_valid      = next_stage_valid;
            exp_stage_data       = next_stage_data;
            exp_stage_data_known = next_stage_data_known;
            exp_out_valid        = next_out_valid;
            exp_out_data         = next_out_data;
            exp_out_data_known   = next_out_data_known;
            checks = checks + 1;
        end
    endtask

    task automatic write_word;
        input [ADDR_W-1:0] addr;
        input [511:0] data;
        begin
            run_cycle(1'b1, 1'b1, 1'b1, addr, data,
                      1'b0, {ADDR_W{1'b0}});
        end
    endtask

    task automatic read_word;
        input [ADDR_W-1:0] addr;
        begin
            run_cycle(1'b1, 1'b1, 1'b0, {ADDR_W{1'b0}}, 512'd0,
                      1'b1, addr);
        end
    endtask

    task automatic idle_cycle;
        begin
            run_cycle(1'b1, 1'b1, 1'b0, {ADDR_W{1'b0}}, 512'd0,
                      1'b0, {ADDR_W{1'b0}});
        end
    endtask

    initial begin
        if ((RD_LAT != 1) && (RD_LAT != 2))
            $fatal(1, "unsupported RD_LAT=%0d", RD_LAT);

        clk_i = 1'b0;
        rst_ni = 1'b0;
        ce_i = 1'b0;
        wr_en_i = 1'b0;
        wr_addr_i = {ADDR_W{1'b0}};
        wr_data_i = 512'd0;
        rd_en_i = 1'b0;
        rd_addr_i = {ADDR_W{1'b0}};

        exp_stage_valid = 1'b0;
        exp_stage_data = 512'd0;
        exp_stage_data_known = 1'b0;
        exp_out_valid = 1'b0;
        exp_out_data = 512'd0;
        exp_out_data_known = 1'b0;

        checks = 0;
        accepted_reads = 0;
        accepted_writes = 0;
        concurrent_ops = 0;
        ce_stalls = 0;
        collision_adjustments = 0;

        for (idx = 0; idx < DEPTH; idx = idx + 1)
            model_known[idx] = 1'b0;

        if (!$value$plusargs("SEED=%d", seed))
            seed = 1;
        rng_seed = seed;
        seed_junk = $urandom(rng_seed);

        // Reset dominates either CE value and clears read-valid state only.
        run_cycle(1'b0, 1'b1, 1'b0, 9'd0, 512'd0, 1'b0, 9'd0);
        run_cycle(1'b0, 1'b0, 1'b1, 9'd1, patterned_word(32'hbad00001),
                  1'b1, 9'd2);
        idle_cycle();

        // Boundary addresses and back-to-back (II=1) accepted reads.
        word_zero = patterned_word(32'h01234567);
        word_last = patterned_word(32'h89abcdef);
        write_word(9'd0, word_zero);
        write_word(DEPTH-1, word_last);
        read_word(9'd0);
        read_word(DEPTH-1);
        idle_cycle();
        idle_cycle();

        // Initialize a known region and exercise sustained II=1 reads.
        for (idx = 0; idx < 64; idx = idx + 1)
            write_word(9'd64 + idx[8:0],
                       patterned_word(32'h10000000 + idx));
        for (idx = 0; idx < 64; idx = idx + 1)
            read_word(9'd64 + idx[8:0]);
        idle_cycle();
        idle_cycle();

        // Legal same-cycle 1R1W always uses different addresses.
        for (idx = 0; idx < 16; idx = idx + 1) begin
            word_tmp = patterned_word(32'h20000000 + idx);
            run_cycle(1'b1, 1'b1,
                      1'b1, 9'd160 + idx[8:0], word_tmp,
                      1'b1, 9'd64 + idx[8:0]);
        end
        for (idx = 0; idx < 16; idx = idx + 1)
            read_word(9'd160 + idx[8:0]);
        idle_cycle();
        idle_cycle();

        // A CE stall freezes the response and rejects both requested ops.
        read_word(9'd0);
        if (RD_LAT == 2)
            idle_cycle();
        run_cycle(1'b1, 1'b0, 1'b1, 9'd0,
                  patterned_word(32'hdeadc0de), 1'b1, 9'd64);
        run_cycle(1'b1, 1'b0, 1'b1, 9'd0,
                  patterned_word(32'hdeadc0df), 1'b1, 9'd65);
        run_cycle(1'b1, 1'b0, 1'b0, 9'd0, 512'd0, 1'b0, 9'd0);
        idle_cycle();
        read_word(9'd0);
        idle_cycle();
        idle_cycle();

        // Memory and data storage survive reset; reset-time write is rejected.
        word_reset = patterned_word(32'h55aa00ff);
        write_word(9'd511, word_reset);
        read_word(9'd511);
        idle_cycle();
        run_cycle(1'b0, 1'b1, 1'b1, 9'd511,
                  patterned_word(32'ha5a55a5a), 1'b1, 9'd64);
        run_cycle(1'b0, 1'b0, 1'b1, 9'd510,
                  patterned_word(32'hfacecafe), 1'b1, 9'd65);
        idle_cycle();
        read_word(9'd511);
        idle_cycle();
        idle_cycle();

        // Seeded random traffic over an initialized address window.
        for (idx = 0; idx < 64; idx = idx + 1)
            write_word(9'd256 + idx[8:0], random_word());

        for (idx = 0; idx < 300; idx = idx + 1) begin
            rand_ce = ($urandom_range(3, 0) != 0);
            rand_wr = ($urandom_range(1, 0) != 0);
            rand_rd = ($urandom_range(1, 0) != 0);
            rand_wr_addr = 9'd256 + $urandom_range(63, 0);
            rand_rd_addr = 9'd256 + $urandom_range(63, 0);

            if (rand_ce && rand_wr && rand_rd &&
                (rand_wr_addr == rand_rd_addr)) begin
                if (rand_rd_addr == 9'd319)
                    rand_rd_addr = 9'd256;
                else
                    rand_rd_addr = rand_rd_addr + 1'b1;
                collision_adjustments = collision_adjustments + 1;
            end

            word_tmp = random_word();
            run_cycle(1'b1, rand_ce, rand_wr, rand_wr_addr, word_tmp,
                      rand_rd, rand_rd_addr);
        end

        idle_cycle();
        idle_cycle();

        // Read back every random-window location to cover accepted writes.
        for (idx = 0; idx < 64; idx = idx + 1)
            read_word(9'd256 + idx[8:0]);
        idle_cycle();
        idle_cycle();

        if (accepted_reads < 150)
            $fatal(1, "insufficient accepted reads: %0d", accepted_reads);
        if (accepted_writes < 120)
            $fatal(1, "insufficient accepted writes: %0d", accepted_writes);
        if (concurrent_ops < 16)
            $fatal(1, "insufficient different-address concurrent ops: %0d",
                   concurrent_ops);
        if (ce_stalls < 3)
            $fatal(1, "CE stall coverage missing");

        $display("[PASS] tb_vfu_spad_mem16 RD_LAT=%0d SEED=%0d checks=%0d reads=%0d writes=%0d concurrent=%0d stalls=%0d collision_adjustments=%0d",
                 RD_LAT, seed, checks, accepted_reads, accepted_writes,
                 concurrent_ops, ce_stalls, collision_adjustments);
        $finish;
    end

endmodule
