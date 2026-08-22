`timescale 1ns/1ps
`include "vfu_defs.vh"

module tb_vfu_post_alu_lane;

    reg [3:0] op_i;
    reg signed [47:0] p_i;
    reg [5:0] shift_i;
    reg [1:0] range_i;
    reg [7:0] low_code_i;
    reg [7:0] high_code_i;
    reg pair_valid_i;
    reg force_zero_i;
    reg signed [7:0] skip_i;

    wire [31:0] data_o;
    wire signed [15:0] moment_s_o;
    wire [22:0] moment_q_o;

    vfu_post_alu_lane dut (
        .op_i(op_i),
        .p_i(p_i),
        .shift_i(shift_i),
        .range_i(range_i),
        .low_code_i(low_code_i),
        .high_code_i(high_code_i),
        .pair_valid_i(pair_valid_i),
        .force_zero_i(force_zero_i),
        .skip_i(skip_i),
        .data_o(data_o),
        .moment_s_o(moment_s_o),
        .moment_q_o(moment_q_o)
    );

    task defaults;
        begin
            op_i = `VFU_OP_RQ;
            p_i = 48'sd0;
            shift_i = 6'd0;
            range_i = 2'b00;
            low_code_i = 8'd0;
            high_code_i = 8'd0;
            pair_valid_i = 1'b1;
            force_zero_i = 1'b0;
            skip_i = 8'sd0;
        end
    endtask

    task expect_s32;
        input signed [31:0] expected;
        begin
            #1;
            if ($signed(data_o) !== expected) begin
                $display("FAIL op=%h p=%0d shift=%0d got=%0d expected=%0d",
                         op_i, $signed(p_i), shift_i,
                         $signed(data_o), expected);
                $fatal(1);
            end
        end
    endtask

    initial begin
        defaults();

        // RNE ties-to-even: +2.5 -> +2, +3.5 -> +4
        op_i = `VFU_OP_RQ;
        shift_i = 1;

        p_i = 5;
        expect_s32(2);

        p_i = 7;
        expect_s32(4);

        // Signed ties-to-even: -2.5 -> -2, -3.5 -> -4
        p_i = -5;
        expect_s32(-2);

        p_i = -7;
        expect_s32(-4);

        // Narrow-S8 forbids -128.
        shift_i = 0;
        p_i = -200;
        expect_s32(-127);

        // RQ_RES: main8=100, skip=100 -> raw S9 200, no saturation.
        op_i = `VFU_OP_RQ_RES;
        p_i = 100;
        skip_i = 100;
        expect_s32(200);

        // QEXP tail then mask.
        op_i = `VFU_OP_QEXP;
        range_i = 2'b10;
        high_code_i = 127;
        pair_valid_i = 1;
        expect_s32(127);

        pair_valid_i = 0;
        expect_s32(0);

        // SM_RECIP semantic zero override.
        defaults();
        op_i = `VFU_OP_SM_RECIP_RAW;
        shift_i = 8;
        p_i = 48'sd123456;
        force_zero_i = 1;
        expect_s32(0);

        // LN_NORM raw S25 bypass.
        defaults();
        op_i = `VFU_OP_LN_NORM;
        p_i = -48'sd123456;
        shift_i = 31; // ignored
        expect_s32(-123456);

        // Moment split: P=(S<<23)+Q.
        defaults();
        op_i = `VFU_OP_LN_MOMENT_ACC;
        p_i = (-48'sd123 <<< 23) + 48'sd4567;
        #1;
        if ($signed(moment_s_o) !== -16'sd123 || moment_q_o !== 23'd4567) begin
            $display("FAIL Moment split S=%0d Q=%0d",
                     $signed(moment_s_o), moment_q_o);
            $fatal(1);
        end

        $display("PASS: vfu_post_alu_lane directed tests");
        $finish;
    end

endmodule
