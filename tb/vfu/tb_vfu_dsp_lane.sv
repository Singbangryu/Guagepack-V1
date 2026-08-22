`timescale 1ns/1ps
`include "vfu_defs.vh"

module tb_vfu_dsp_lane;

    reg clk = 1'b0;
    always #2.5 clk = ~clk; // 200 MHz

    reg rst_n = 1'b0;
    reg ce = 1'b1;
    reg valid = 1'b0;
    reg [3:0] op = 4'd0;
    reg signed [26:0] a = 27'sd0;
    reg signed [17:0] b = 18'sd0;
    reg signed [47:0] c = 48'sd0;

    wire valid_o;
    wire [3:0] op_o;
    wire signed [47:0] p_o;

    vfu_dsp_lane dut (
        .clk_i(clk),
        .rst_ni(rst_n),
        .ce_i(ce),
        .valid_i(valid),
        .op_i(op),
        .a_i(a),
        .b_i(b),
        .c_i(c),
        .valid_o(valid_o),
        .op_o(op_o),
        .p_o(p_o)
    );

    task push;
        input [3:0] t_op;
        input signed [26:0] t_a;
        input signed [17:0] t_b;
        input signed [47:0] t_c;
        begin
            @(negedge clk);
            valid = 1'b1;
            op = t_op;
            a = t_a;
            b = t_b;
            c = t_c;
        end
    endtask

    task bubble;
        begin
            @(negedge clk);
            valid = 1'b0;
            op = 4'd0;
            a = 27'sd0;
            b = 18'sd0;
            c = 48'sd0;
        end
    endtask

    task expect_p;
        input signed [47:0] expected;
        input [3:0] expected_op;
        begin
            while (!valid_o)
                @(posedge clk);
            #1;
            if ($signed(p_o) !== expected) begin
                $display("FAIL op=%h got=%0d expected=%0d",
                         op_o, $signed(p_o), expected);
                $fatal(1);
            end
            if (op_o !== expected_op) begin
                $display("FAIL op tag got=%h expected=%h", op_o, expected_op);
                $fatal(1);
            end
            $display("PASS op=%h p=%0d", op_o, $signed(p_o));
            @(posedge clk);
        end
    endtask

    initial begin
        repeat (3) @(posedge clk);
        rst_n = 1'b1;

        // ---------------------------------------------------------------------
        // 1) Normal: C + A*B = 100 + 7*(-3) = 79
        // ---------------------------------------------------------------------
        push(`VFU_OP_GELU, 27'sd7, -18'sd3, 48'sd100);
        bubble();
        expect_p(48'sd79, `VFU_OP_GELU);

        // ---------------------------------------------------------------------
        // 2) LN_D: C + A*B, with B=-S
        //    S=-1234, Q=50000 -> C=Q<<7=6,400,000
        //    D=6,400,000 + (-1234 * +1234) = 4,877,244
        // ---------------------------------------------------------------------
        push(`VFU_OP_LN_D, -27'sd1234, 18'sd1234, 48'sd6400000);
        bubble();
        expect_p(48'sd4877244, `VFU_OP_LN_D);

        // ---------------------------------------------------------------------
        // 3) Moment INIT then back-to-back ACC.
        //    z=+5:
        //      A=2^23+5=8,388,613, B=5
        //      P0=41,943,065
        //    z=-2:
        //      A=2^23-2=8,388,606, B=-2
        //      P1=P0-16,777,212=25,165,853
        // ---------------------------------------------------------------------
        push(`VFU_OP_LN_MOMENT_INIT, 27'sd8388613, 18'sd5, 48'sd0);
        push(`VFU_OP_LN_MOMENT_ACC,  27'sd8388606, -18'sd2, 48'sd0);
        bubble();

        expect_p(48'sd41943065, `VFU_OP_LN_MOMENT_INIT);
        expect_p(48'sd25165853, `VFU_OP_LN_MOMENT_ACC);

        // ---------------------------------------------------------------------
        // 4) Bubble must NOT re-accumulate stale moment operands.
        //    P must hold until a valid item arrives.
        // ---------------------------------------------------------------------
        repeat (3) bubble();

        // New INIT overwrites stale P, proving no explicit RSTP is required
        // at a row boundary.
        push(`VFU_OP_LN_MOMENT_INIT, 27'sd8388608, 18'sd0, 48'sd0);
        bubble();
        expect_p(48'sd0, `VFU_OP_LN_MOMENT_INIT);

        $display("ALL DSP LANE TESTS PASSED");
        $finish;
    end

endmodule
