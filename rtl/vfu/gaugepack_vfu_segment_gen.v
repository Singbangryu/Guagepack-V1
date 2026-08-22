`timescale 1ns/1ps

module gaugepack_vfu_segment_gen (
    input  wire signed [31:0] x_i,
    input  wire [404:0] boundary_flat_i,
    input  wire signed [26:0] x_min_i,
    input  wire signed [26:0] x_max_i,
    output wire [3:0] addr_o,
    output wire [1:0] range_o
);
    localparam [1:0] RANGE_MIDDLE = 2'b00;
    localparam [1:0] RANGE_LOW    = 2'b01;
    localparam [1:0] RANGE_HIGH   = 2'b10;

    wire signed [31:0] x_min_32_w = {{5{x_min_i[26]}}, x_min_i};
    wire signed [31:0] x_max_32_w = {{5{x_max_i[26]}}, x_max_i};
    wire signed [26:0] x27_w = x_i[26:0];
    wire signed [26:0] boundary_w [0:14];
    wire [14:0] ge_w;
    wire [3:0] seg_w;

    assign range_o =
        (x_i < x_min_32_w) ? RANGE_LOW :
        (x_i > x_max_32_w) ? RANGE_HIGH :
                             RANGE_MIDDLE;

    genvar g;
    generate
        for (g = 0; g < 15; g = g + 1) begin : GEN_BOUNDARY_CMP
            assign boundary_w[g] = boundary_flat_i[g*27 +: 27];
            assign ge_w[g] = ($signed(x27_w) >= $signed(boundary_w[g]));
        end
    endgenerate

    assign seg_w[3] = ge_w[7];
    assign seg_w[2] = (ge_w[3] & ~ge_w[7]) | ge_w[11];
    assign seg_w[1] = (ge_w[1] & ~ge_w[3])
                    | (ge_w[5] & ~ge_w[7])
                    | (ge_w[9] & ~ge_w[11])
                    | ge_w[13];
    assign seg_w[0] = (ge_w[0] & ~ge_w[1])
                    | (ge_w[2] & ~ge_w[3])
                    | (ge_w[4] & ~ge_w[5])
                    | (ge_w[6] & ~ge_w[7])
                    | (ge_w[8] & ~ge_w[9])
                    | (ge_w[10] & ~ge_w[11])
                    | (ge_w[12] & ~ge_w[13])
                    | ge_w[14];

    assign addr_o = seg_w;
endmodule
