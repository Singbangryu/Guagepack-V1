`timescale 1ns/1ps
`include "vfu_internal_op_defs.vh"

// Refactor equivalence test. Compile the new stages, the two archived lane
// modules in rtl/vfu/vfu_draft, and the simulator's DSP48E2/glbl library.
// The archived POST reference uses force_zero_i=0, matching the CORE contract.
module tb_vfu_stage16;
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst_n = 1'b0;

    reg dsp_ce = 1'b1;
    reg dsp_valid = 1'b0;
    reg [3:0] dsp_op = 4'hf;
    reg [431:0] a = 432'd0;
    reg [287:0] b = 288'd0;
    reg [767:0] c = 768'd0;
    wire dsp_valid_new;
    wire [3:0] dsp_op_new;
    wire [767:0] dsp_p_new;
    wire [15:0] dsp_valid_ref;
    wire [63:0] dsp_op_ref;
    wire [767:0] dsp_p_ref;

    vfu_dsp16 u_dsp16 (
        .clk_i(clk), .rst_ni(rst_n), .ce_i(dsp_ce),
        .valid_i(dsp_valid), .op_i(dsp_op), .a_i(a), .b_i(b), .c_i(c),
        .valid_o(dsp_valid_new), .op_o(dsp_op_new), .p_o(dsp_p_new)
    );

    reg post_valid = 1'b0;
    reg [3:0] post_op = 4'hf;
    reg [8:0] feature = 9'd0;
    reg [3:0] tile = 4'd0;
    reg [15:0] mask = 16'd0;
    reg key_valid = 1'b0, last = 1'b0;
    reg [767:0] post_p = 768'd0;
    reg [95:0] shamt = 96'd0;
    reg [31:0] range_code = 32'd0;
    reg [7:0] low_code = 8'd0, high_code = 8'd0;
    reg [127:0] skip = 128'd0;

    wire post_valid_new;
    wire [3:0] post_op_new;
    wire [8:0] feature_new;
    wire [3:0] tile_new;
    wire [15:0] mask_new;
    wire key_new, last_new;
    wire [511:0] data_new;
    wire capture_new;
    wire [255:0] s_new;
    wire [367:0] q_new;

    vfu_post_alu16_s3 u_post16 (
        .clk_i(clk), .rst_ni(rst_n), .valid_i(post_valid), .op_i(post_op),
        .feature_i(feature), .token_tile_i(tile), .lane_valid_i(mask),
        .key_valid_i(key_valid), .last_i(last), .p_i(post_p),
        .shamt_i(shamt), .range_i(range_code), .low_code_i(low_code),
        .high_code_i(high_code), .skip_i(skip),
        .valid_o(post_valid_new), .op_o(post_op_new), .feature_o(feature_new),
        .token_tile_o(tile_new), .lane_valid_o(mask_new), .key_valid_o(key_new),
        .last_o(last_new), .data_o(data_new), .moment_capture_o(capture_new),
        .moment_s_o(s_new), .moment_q_o(q_new)
    );

    wire [511:0] post_data_ref;
    wire [255:0] post_s_ref;
    wire [367:0] post_q_ref;

    genvar lane;
    generate
        for (lane = 0; lane < 16; lane = lane + 1) begin : GEN_REFERENCE
            vfu_dsp_lane u_dsp_lane (
                .clk_i(clk), .rst_ni(rst_n), .ce_i(dsp_ce),
                .valid_i(dsp_valid), .op_i(dsp_op),
                .a_i($signed(a[lane*27 +: 27])),
                .b_i($signed(b[lane*18 +: 18])),
                .c_i($signed(c[lane*48 +: 48])),
                .valid_o(dsp_valid_ref[lane]),
                .op_o(dsp_op_ref[lane*4 +: 4]),
                .p_o(dsp_p_ref[lane*48 +: 48])
            );
            vfu_post_alu_lane u_post_lane (
                .op_i(post_op), .p_i($signed(post_p[lane*48 +: 48])),
                .shamt_i(shamt[lane*6 +: 6]), .range_i(range_code[lane*2 +: 2]),
                .low_code_i(low_code), .high_code_i(high_code),
                .key_valid_i(key_valid), .force_zero_i(1'b0),
                .skip_i($signed(skip[lane*8 +: 8])),
                .data_o(post_data_ref[lane*32 +: 32]),
                .moment_s_o(post_s_ref[lane*16 +: 16]),
                .moment_q_o(post_q_ref[lane*23 +: 23])
            );
        end
    endgenerate

    // Reference S3 boundary around the archived combinational POST lanes.
    reg expected_valid;
    reg [3:0] expected_op;
    reg [30:0] expected_meta;
    reg [511:0] expected_data;
    reg expected_capture;
    reg [255:0] expected_s;
    reg [367:0] expected_q;
    wire capture = post_valid && last &&
        ((post_op == `VFU_OP_LN_MOMENT_INIT) ||
         (post_op == `VFU_OP_LN_MOMENT_ACC));

    always @(posedge clk) begin
        if (!rst_n) begin
            expected_valid <= 0;
            expected_op <= 0;
            expected_meta <= 0;
            expected_data <= 0;
            expected_capture <= 0;
            expected_s <= 0;
            expected_q <= 0;
        end else begin
            expected_valid <= post_valid;
            expected_capture <= capture;
            if (post_valid) begin
                expected_op <= post_op;
                expected_meta <= {feature, tile, mask, key_valid, last};
                expected_data <= post_data_ref;
            end
            if (capture) begin
                expected_s <= post_s_ref;
                expected_q <= post_q_ref;
            end
        end
    end

    task check_outputs;
        integer l;
        begin
            for (l = 0; l < 16; l = l + 1) begin
                if (dsp_valid_new !== dsp_valid_ref[l] ||
                    dsp_op_new !== dsp_op_ref[l*4 +: 4] ||
                    dsp_p_new[l*48 +: 48] !== dsp_p_ref[l*48 +: 48])
                    $fatal(1, "DSP16 mismatch at lane %0d, time %0t", l, $time);
            end
            if (post_valid_new !== expected_valid || post_op_new !== expected_op ||
                {feature_new,tile_new,mask_new,key_new,last_new} !== expected_meta ||
                data_new !== expected_data || capture_new !== expected_capture ||
                s_new !== expected_s || q_new !== expected_q)
                $fatal(1, "POST16/S3 mismatch at time %0t", $time);
        end
    endtask

    integer cycle, l;
    integer seed = 32'h16060906;
    initial begin
        // Let the vendor model finish its natural GSR interval.
        wait (glbl.GSR === 1'b0);
        repeat (3) @(negedge clk);
        rst_n = 1'b1;
        for (cycle = 0; cycle < 512; cycle = cycle + 1) begin
            rst_n = (cycle % 137 != 90);
            dsp_ce = (cycle % 7 != 3);
            dsp_valid = (cycle % 5 != 2);
            dsp_op = cycle % 12;
            post_valid = (cycle % 4 != 2);
            post_op = (cycle + 3) % 12;
            feature = cycle;
            tile = cycle % 4;
            mask = (cycle % 3 == 0) ? 16'hffff :
                   (cycle % 3 == 1) ? 16'h001f : 16'h0000;
            key_valid = (cycle % 3 != 0);
            last = (cycle % 5 == 0);
            low_code = $random(seed);
            high_code = $random(seed);
            for (l = 0; l < 16; l = l + 1) begin
                a[l*27 +: 27] = $random(seed);
                b[l*18 +: 18] = $random(seed);
                c[l*48 +: 48] = {$random(seed), $random(seed)};
                post_p[l*48 +: 48] = {$random(seed), $random(seed)};
                shamt[l*6 +: 6] = (cycle + l) % 64;
                range_code[l*2 +: 2] = (cycle + l) % 4;
                skip[l*8 +: 8] = $random(seed);
            end
            @(posedge clk); #1;
            check_outputs;
            @(negedge clk);
        end
        rst_n = 1'b1;
        dsp_ce = 1'b1;
        dsp_valid = 1'b0;
        post_valid = 1'b0;
        repeat (4) begin
            @(posedge clk); #1;
            check_outputs;
            @(negedge clk);
        end
        $display("PASS: DSP16 and POST16/S3 match 16 archived lanes over 512 cycles plus drain");
        $finish;
    end
    initial begin
        #20000;
        $fatal(1, "timeout");
    end
endmodule
