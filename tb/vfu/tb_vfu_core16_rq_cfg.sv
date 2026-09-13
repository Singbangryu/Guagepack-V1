`timescale 1ns/1ps
`include "vfu_internal_op_defs.vh"

// Focused CORE test. Compile with the real PRE/DSP/POST and vendor DSP48E2/glbl,
// excluding the empty production vfu_coeff_page16.v (test provider below).
module tb_vfu_core16_rq_cfg;
    reg clk = 0;
    always #5 clk = ~clk;
    reg rst_n = 0, valid = 0;
    reg [3:0] op = 4'hf; // Idle value; first command establishes the decode.
    reg [8:0] feature = 0;
    reg [3:0] tile = 0;
    reg [15:0] mask = 0;
    reg key_valid = 0, last = 0;
    reg [511:0] src0 = 0, src1 = 0;
    reg [127:0] skip_for_input = 0, skip_s1 = 0;
    reg signed [17:0] req_mult = 0;
    reg [5:0] req_shamt = 0;
    reg [7:0] page = 0;
    wire s0_valid, out_valid, out_key, out_last, moment_capture;
    wire [3:0] s0_op, s0_tile, out_op, out_tile;
    wire [8:0] s0_feature, out_feature;
    wire [15:0] out_mask;
    wire [511:0] out_data;
    wire [255:0] moment_s;
    wire [367:0] moment_q;

    vfu_core16 dut (
        .clk_i(clk), .rst_ni(rst_n), .valid_i(valid), .op_i(op),
        .feature_i(feature), .token_tile_i(tile), .lane_valid_i(mask),
        .key_valid_i(key_valid), .last_i(last), .src0_i(src0), .src1_i(src1),
        .rsqrted_i(128'd0), .page_id_i(page),
        .req_mult_i(req_mult), .req_shamt_i(req_shamt), .skip_s1_i(skip_s1),
        .s0_valid_o(s0_valid), .s0_op_o(s0_op), .s0_feature_o(s0_feature),
        .s0_token_tile_o(s0_tile), .valid_o(out_valid), .op_o(out_op),
        .feature_o(out_feature), .token_tile_o(out_tile), .lane_valid_o(out_mask),
        .key_valid_o(out_key), .last_o(out_last), .data_o(out_data),
        .moment_capture_o(moment_capture), .moment_s_o(moment_s), .moment_q_o(moment_q)
    );

    // Arithmetic reference uses unsigned magnitude/division, independently of
    // the production signed floor/guard/sticky RNE implementation.
    function automatic signed [63:0] rne(input signed [63:0] p, input integer f);
        reg [63:0] mag, q, rem_bits, half;
        begin
            mag = (p < 0) ? -p : p;
            q = mag >> f;
            if (f != 0) begin
                rem_bits = mag & ((64'd1 << f) - 1);
                half = 64'd1 << (f - 1);
                if (rem_bits > half || (rem_bits == half && q[0])) q = q + 1;
            end
            rne = (p < 0) ? -$signed(q) : $signed(q);
        end
    endfunction

    function automatic signed [31:0] clamp_s8(input signed [63:0] x);
        begin
            clamp_s8 = (x > 127) ? 127 : (x < -127) ? -127 : x;
        end
    endfunction

    function automatic signed [31:0] sample(input integer index, input integer family);
        begin
            if (family == 1) begin
                case (index % 4)
                    0: sample = 67108863;
                    1: sample = -67108864;
                    2: sample = 65537;
                    3: sample = -65537;
                endcase
            end else if (family == 2) sample = index - 8;
            else begin
                case (index % 16)
                    0: sample = -257;  1: sample = -255;
                    2: sample = -253;  3: sample = -5;
                    4: sample = -3;    5: sample = -1;
                    6: sample = 0;     7: sample = 1;
                    8: sample = 3;     9: sample = 5;
                    10: sample = 253;  11: sample = 255;
                    12: sample = 257;  13: sample = 127;
                    14: sample = -127; 15: sample = 2;
                endcase
            end
        end
    endfunction

    reg exp_valid [0:3];
    reg [3:0] exp_op [0:3];
    reg [30:0] exp_meta [0:3];
    reg [511:0] exp_data [0:3];
    reg [127:0] skip_delay [0:1];
    integer commands = 0, sent = 0, received = 0, lane_checks = 0;
    integer bubble_checks = 0, consecutive = 0, pos_ties = 0, neg_ties = 0;
    integer high_clamps = 0, low_clamps = 0, residual_wide = 0, nonrq_checks = 0;
    reg previous_valid = 0;
    integer stage, lane, shift;
    reg signed [63:0] x, m, c, p, rounded, result_lane;
    reg [63:0] magnitude, remainder_bits;

    // An input accepted at edge n needs its skip on skip_s1_i before edge n+2.
    // This transport is test-owned and driven from input history, not DUT state.
    always @(negedge clk) skip_s1 = skip_delay[1];

    always @(posedge clk) begin
        if (!rst_n) begin
            for (stage = 0; stage < 4; stage = stage + 1) begin
                exp_valid[stage] = 0;
                exp_op[stage] = 0;
                exp_meta[stage] = 0;
                exp_data[stage] = 0;
            end
            skip_delay[0] = 0;
            skip_delay[1] = 0;
            previous_valid = 0;
        end else begin
            for (stage = 3; stage > 0; stage = stage - 1) begin
                exp_valid[stage] = exp_valid[stage-1];
                exp_op[stage] = exp_op[stage-1];
                exp_meta[stage] = exp_meta[stage-1];
                exp_data[stage] = exp_data[stage-1];
            end
            skip_delay[1] = skip_delay[0];
            skip_delay[0] = skip_for_input;
            exp_valid[0] = valid;
            exp_op[0] = op;
            exp_meta[0] = {feature, tile, mask, key_valid, last};
            if (valid) begin
                sent = sent + 1;
                if (previous_valid) consecutive = consecutive + 1;
                for (lane = 0; lane < 16; lane = lane + 1) begin
                    x = $signed(src0[lane*32 +: 32]);
                    if (op == `VFU_OP_RQ || op == `VFU_OP_RQ_RES) begin
                        m = $signed(req_mult);
                        c = 0;
                        shift = req_shamt;
                    end else if (op == `VFU_OP_SM_CONTEXT) begin
                        m = $signed(src1[lane*32 +: 32]);
                        c = 0;
                        shift = 23;
                    end else begin
                        // Synthetic page's public test formula, not DUT reads.
                        m = (lane % 2) ? -(lane + 1) : lane + 1;
                        c = 8 * (feature % 7 - 3) + 4 * lane + page;
                        shift = lane % 3 + 1;
                    end
                    p = x * m + c;
                    rounded = rne(p, shift);
                    result_lane = clamp_s8(rounded);
                    if (op == `VFU_OP_RQ || op == `VFU_OP_RQ_RES) begin
                        if (rounded > 127) high_clamps = high_clamps + 1;
                        if (rounded < -127) low_clamps = low_clamps + 1;
                        if (shift > 0) begin
                            magnitude = (p < 0) ? -p : p;
                            remainder_bits = magnitude & ((64'd1 << shift) - 1);
                            if (remainder_bits == (64'd1 << (shift - 1))) begin
                                if (p < 0) neg_ties = neg_ties + 1;
                                else pos_ties = pos_ties + 1;
                            end
                        end
                    end else nonrq_checks = nonrq_checks + 1;
                    if (op == `VFU_OP_RQ_RES) begin
                        result_lane = result_lane + $signed(skip_for_input[lane*8 +: 8]);
                        if (result_lane > 127 || result_lane < -127)
                            residual_wide = residual_wide + 1;
                    end
                    exp_data[0][lane*32 +: 32] = result_lane;
                end
            end
            previous_valid = valid;
        end
        #1;
        if (s0_valid !== exp_valid[0] || out_valid !== exp_valid[3])
            $fatal(1, "valid/latency mismatch at %0t", $time);
        if (s0_valid && {s0_op,s0_feature,s0_tile} !==
                        {exp_op[0],exp_meta[0][30:18]})
            $fatal(1, "S0 metadata mismatch at %0t", $time);
        if (moment_capture !== 1'b0 || moment_s !== 256'd0 || moment_q !== 368'd0)
            $fatal(1, "unexpected Moment state at %0t", $time);
        if (out_valid) begin
            if ({out_op,out_feature,out_tile,out_mask,out_key,out_last} !==
                {exp_op[3],exp_meta[3]})
                $fatal(1, "S3 metadata/order/last mismatch at %0t", $time);
            for (lane = 0; lane < 16; lane = lane + 1) begin
                if (out_data[lane*32 +: 32] !== exp_data[3][lane*32 +: 32])
                    $fatal(1, "data mismatch beat=%0d op=%0h lane=%0d got=%0d expected=%0d",
                           received, out_op, lane, $signed(out_data[lane*32 +: 32]),
                           $signed(exp_data[3][lane*32 +: 32]));
                lane_checks = lane_checks + 1;
            end
            received = received + 1;
        end else if (rst_n) bubble_checks = bubble_checks + 1;
    end

    task automatic beat(input integer n, input integer family, input bit is_last);
        integer l;
        begin
            @(negedge clk);
            valid = 1;
            feature = 17 * n + 3;
            tile = n % 4;
            mask = (n == 0) ? 16'hffff : (n == 1) ? 16'h0025 :
                   (n == 2) ? 16'h0000 : 16'ha55a;
            key_valid = n % 2;
            last = is_last;
            for (l = 0; l < 16; l = l + 1) begin
                src0[l*32 +: 32] = sample((l + n) % 16, family);
                src1[l*32 +: 32] = 65536 + l;
                skip_for_input[l*8 +: 8] = ((l + n) % 2) ? -127 : 127;
            end
            @(posedge clk); #2;
        end
    endtask

    task automatic bubble;
        begin
            @(negedge clk);
            valid = 0;
            src0 = 'x;
            src1 = 'x;
            skip_for_input = 'x;
            last = 0;
            @(posedge clk); #2;
        end
    endtask

    task automatic command(input [3:0] next_op, input signed [17:0] next_m,
                           input [5:0] next_f, input integer family, input [7:0] next_page);
        begin
            // Previous command always completes this drain before returning.
            if (sent != received) $fatal(1, "configuration changed before drain");
            @(negedge clk);
            op = next_op;
            req_mult = next_m;
            req_shamt = next_f;
            page = next_page;
            commands = commands + 1;
            beat(0, family, 0);
            beat(1, family, 0);
            bubble;
            beat(2, family, 0);
            beat(3, family, 1);
            repeat (4) bubble;
        end
    endtask

    initial begin
        wait (glbl.GSR === 1'b0);
        repeat (3) @(negedge clk);
        rst_n = 1;
        command(`VFU_OP_RQ, 1, 1, 0, 0);
        command(`VFU_OP_RQ, -1, 1, 0, 1);
        command(`VFU_OP_RQ, 1, 0, 0, 0);
        command(`VFU_OP_RQ, 3, 2, 0, 1);
        command(`VFU_OP_RQ, -3, 2, 0, 0);
        command(`VFU_OP_RQ, 0, 0, 1, 1);
        command(`VFU_OP_RQ, 131071, 17, 0, 0);
        command(`VFU_OP_RQ, -131072, 17, 0, 1);
        command(`VFU_OP_RQ, 131071, 31, 1, 0);
        command(`VFU_OP_RQ, -131072, 47, 1, 1);
        command(`VFU_OP_RQ, 131071, 48, 1, 0);
        command(`VFU_OP_RQ, -131072, 63, 1, 1);
        command(`VFU_OP_RQ_RES, 1, 0, 0, 0);
        command(`VFU_OP_RQ_RES, 1, 1, 0, 1);
        command(`VFU_OP_RQ_RES, -3, 2, 0, 0);
        command(`VFU_OP_RQ_RES, -131072, 17, 0, 1);
        // Repeat identical non-RQ inputs with radically different external M/F;
        // X config must also be ignored, and page C must remain effective.
        command(`VFU_OP_LN_AFFINE, 131071, 0, 2, 7);
        command(`VFU_OP_LN_AFFINE, -131072, 63, 2, 7);
        command(`VFU_OP_LN_AFFINE, 18'bx, 6'bx, 2, 7);
        command(`VFU_OP_GELU, 18'bx, 6'bx, 2, 11);
        command(`VFU_OP_SM_CONTEXT, 18'bx, 6'bx, 0, 23);
        if (sent != received || sent != 84 || lane_checks != 1344 ||
            !consecutive || !bubble_checks || !pos_ties || !neg_ties ||
            !high_clamps || !low_clamps || !residual_wide || !nonrq_checks)
            $fatal(1, "incomplete coverage or drain");
        $display("[PASS] CORE16 RQ config: commands=%0d beats=%0d lanes=%0d bubbles=%0d consecutive=%0d",
                 commands, received, lane_checks, bubble_checks, consecutive);
        $display("coverage: positive_ties=%0d negative_ties=%0d high_clamps=%0d low_clamps=%0d wide_residual=%0d nonRQ_lanes=%0d",
                 pos_ties, neg_ties, high_clamps, low_clamps, residual_wide, nonrq_checks);
        $finish;
    end
    initial begin
        #20000;
        $fatal(1, "timeout");
    end
endmodule

// TEST ONLY: replaces the undriven coefficient wrapper during this simulation.
// It is neither a production page implementation nor calibrated coefficients.
// RQ/RQ_RES receive X on every page M/C/F bit; all other page outputs are known.
module vfu_coeff_page16 #(parameter PAGE_ID_W = 8) (
    input wire [PAGE_ID_W-1:0] page_id_i,
    input wire [3:0] op_s0_i,
    input wire [8:0] feature_s0_i,
    input wire [63:0] seg_addr_s0_i,
    output wire [404:0] boundary_flat_o,
    output wire signed [26:0] x_min_o, x_max_o,
    output wire [7:0] low_code_o, high_code_o,
    output reg [287:0] m_o,
    output reg [767:0] c_o,
    output reg [95:0] shamt_o
);
    assign x_min_o = -27'sd1000000;
    assign x_max_o = 27'sd1000000;
    assign low_code_o = 8'h81;
    assign high_code_o = 8'h7f;
    genvar b;
    generate for (b = 0; b < 15; b = b + 1) begin : GEN_BOUNDARY
        assign boundary_flat_o[b*27 +: 27] = (b - 7) * 100;
    end endgenerate
    integer l;
    integer offset;
    always @(*) begin
        offset = 8 * (feature_s0_i % 7 - 3) + page_id_i;
        for (l = 0; l < 16; l = l + 1) begin
            m_o[l*18 +: 18] = (l % 2) ? -(l + 1) : l + 1;
            c_o[l*48 +: 48] = $signed(offset + 4 * l);
            shamt_o[l*6 +: 6] = l % 3 + 1;
        end
        if (op_s0_i == `VFU_OP_RQ || op_s0_i == `VFU_OP_RQ_RES) begin
            m_o = 'x;
            c_o = 'x;
            shamt_o = 'x;
        end
    end
endmodule
