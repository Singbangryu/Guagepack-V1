`timescale 1ns/1ps
`include "vfu_defs.vh"

module tb_vfu_pre_alu16;
    reg clk_i = 1'b0;
    reg rst_ni = 1'b0;
    reg ce_i = 1'b1;
    reg valid_i = 1'b0;
    reg [3:0] op_i = 4'd0;
    reg [8:0] feature_i = 9'd0;
    reg [15:0] lane_valid_i = 16'hFFFF;
    reg [511:0] src0_i = 512'd0;
    reg [511:0] src1_i = 512'd0;
    reg [127:0] rsqrted_i = 128'd0;
    reg [404:0] boundary_flat_i = 405'd0;
    reg signed [26:0] x_min_i = -27'sd1000;
    reg signed [26:0] x_max_i =  27'sd1000;

    wire valid_o;
    wire [3:0] op_o;
    wire [8:0] feature_o;
    wire [15:0] lane_valid_o;
    wire [431:0] a_o;
    wire [287:0] b_o;
    wire [767:0] c_o;
    wire [63:0] seg_addr_o;
    wire [31:0] seg_range_o;

    integer i;
    integer errors = 0;

    always #5 clk_i = ~clk_i;

    vfu_pre_alu16 dut (
        .clk_i(clk_i), .rst_ni(rst_ni), .ce_i(ce_i),
        .valid_i(valid_i), .op_i(op_i), .feature_i(feature_i),
        .lane_valid_i(lane_valid_i), .src0_i(src0_i), .src1_i(src1_i),
        .rsqrted_i(rsqrted_i), .boundary_flat_i(boundary_flat_i),
        .x_min_i(x_min_i), .x_max_i(x_max_i),
        .valid_o(valid_o), .op_o(op_o), .feature_o(feature_o),
        .lane_valid_o(lane_valid_o), .a_o(a_o), .b_o(b_o),
        .c_o(c_o), .seg_addr_o(seg_addr_o), .seg_range_o(seg_range_o)
    );

    task drive_lane0;
        input signed [31:0] s0;
        input signed [31:0] s1;
        input signed [31:0] s2;
        begin
            src0_i = 512'd0;
            src1_i = 512'd0;
            rsqrted_i = 128'd0;
            src0_i[31:0] = s0;
            src1_i[31:0] = s1;
            rsqrted_i[7:0] = s2[7:0];
        end
    endtask

    task pulse;
        begin
            valid_i = 1'b1;
            @(posedge clk_i);
            #1;
            valid_i = 1'b0;
        end
    endtask

    initial begin
        // boundaries = -700,-600,...,+700
        for (i = 0; i < 15; i = i + 1)
            boundary_flat_i[i*27 +: 27] = $signed((i-7)*100);

        repeat (2) @(posedge clk_i);
        rst_ni = 1'b1;

        // QEXP: d = 100 - 140 = -40
        op_i = `VFU_OP_QEXP;
        drive_lane0(32'sd100, 32'sd140, 32'sd0);
        pulse();
        if ($signed(a_o[26:0]) !== -27'sd40) begin
            $display("FAIL QEXP A=%0d", $signed(a_o[26:0])); errors = errors + 1;
        end

        // MomentPack z=+5
        op_i = `VFU_OP_LN_MOMENT_INIT;
        drive_lane0(32'sd5, 0, 0);
        pulse();
        if ($signed(a_o[26:0]) !== 27'sd8388613 || $signed(b_o[17:0]) !== 18'sd5) begin
            $display("FAIL MOM +5 A=%0d B=%0d", $signed(a_o[26:0]), $signed(b_o[17:0])); errors = errors + 1;
        end

        // MomentPack z=-254
        op_i = `VFU_OP_LN_MOMENT_ACC;
        drive_lane0(-32'sd254, 0, 0);
        pulse();
        if ($signed(a_o[26:0]) !== 27'sd8388354 || $signed(b_o[17:0]) !== -18'sd254) begin
            $display("FAIL MOM -254 A=%0d B=%0d", $signed(a_o[26:0]), $signed(b_o[17:0])); errors = errors + 1;
        end

        // LN_D: S=-1234, Q=50000 => B=+1234, C=6,400,000
        op_i = `VFU_OP_LN_D;
        drive_lane0(-32'sd1234, 32'sd50000, 0);
        pulse();
        if ($signed(a_o[26:0]) !== -27'sd1234 ||
            $signed(b_o[17:0]) !== 18'sd1234 ||
            $signed(c_o[47:0]) !== 48'sd6400000) begin
            $display("FAIL LN_D A=%0d B=%0d C=%0d", $signed(a_o[26:0]), $signed(b_o[17:0]), $signed(c_o[47:0])); errors = errors + 1;
        end

        // LN_NORM: z=100, S=1000, rho=127 => n=11800
        op_i = `VFU_OP_LN_NORM;
        drive_lane0(32'sd100, 32'sd1000, 32'sd127);
        pulse();
        if ($signed(a_o[26:0]) !== 27'sd11800 || $signed(b_o[17:0]) !== 18'sd127) begin
            $display("FAIL LN_NORM A=%0d B=%0d", $signed(a_o[26:0]), $signed(b_o[17:0])); errors = errors + 1;
        end

        // SM_CONTEXT: N=-12345, R=54321
        op_i = `VFU_OP_SM_CONTEXT;
        drive_lane0(-32'sd12345, 32'sd54321, 0);
        pulse();
        if ($signed(a_o[26:0]) !== -27'sd12345 || $signed(b_o[17:0]) !== 18'sd54321) begin
            $display("FAIL SM_CONTEXT A=%0d B=%0d", $signed(a_o[26:0]), $signed(b_o[17:0])); errors = errors + 1;
        end

        if (errors == 0) $display("PASS: vfu_pre_alu16 directed tests");
        else             $display("FAIL: vfu_pre_alu16 errors=%0d", errors);
        $finish;
    end
endmodule
