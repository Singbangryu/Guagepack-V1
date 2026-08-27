`timescale 1ns/1ps

// 2-buffer 16x16 S8 V corner-turn storage.
//
// A trusted producer supplies one group as 16 accepted feature columns.  The
// accepted-beat count marks group completion only; in_col_i is the actual
// local column address.  Completed groups drain in FIFO order as token rows.
// The two data matrices are intentionally absent from reset logic.
module vfu_vcorner_ff (
    input  wire         clk_i,
    input  wire         rst_ni,
    input  wire         ce_i,

    // Feature-column input: 16 token lanes for one local feature.
    input  wire         in_valid_i,
    output wire         in_ready_o,
    input  wire [3:0]   in_col_i,
    input  wire [4:0]   in_tag_i,
    input  wire [127:0] in_data_i,

    // Token-row output: 16 feature lanes for one local token/key row.
    output wire         out_valid_o,
    input  wire         out_ready_i,
    output wire [3:0]   out_row_o,
    output wire [4:0]   out_tag_o,
    output wire [127:0] out_data_o,
    output wire         out_last_o,

    output wire         idle_o
);

    // [token row][feature column]
    (* ram_style = "registers" *)
    reg [7:0] bank_0 [0:15][0:15];
    (* ram_style = "registers" *)
    reg [7:0] bank_1 [0:15][0:15];

    reg       bank_0_full_r;
    reg       bank_1_full_r;
    reg [4:0] bank_0_tag_r;
    reg [4:0] bank_1_tag_r;

    reg       write_bank_r;
    reg       read_bank_r;
    reg [3:0] write_col_ptr_r;
    reg [3:0] read_row_ptr_r;

    wire write_bank_full_w;
    wire read_bank_full_w;
    wire input_fire_w;
    wire output_fire_w;

    assign write_bank_full_w = write_bank_r
                             ? bank_1_full_r
                             : bank_0_full_r;

    assign read_bank_full_w = read_bank_r
                            ? bank_1_full_r
                            : bank_0_full_r;

    assign in_ready_o  = ~write_bank_full_w;
    assign out_valid_o = read_bank_full_w;

    assign input_fire_w  = ce_i && in_valid_i && in_ready_o;
    assign output_fire_w = ce_i && out_valid_o && out_ready_i;

    assign out_row_o = read_row_ptr_r;
    assign out_tag_o = read_bank_r ? bank_1_tag_r : bank_0_tag_r;
    assign out_last_o = out_valid_o && (read_row_ptr_r == 4'd15);

    assign idle_o = ~bank_0_full_r &&
                    ~bank_1_full_r &&
                    (write_col_ptr_r == 4'd0);

    genvar feature_idx;
    generate
        for (feature_idx = 0; feature_idx < 16;
             feature_idx = feature_idx + 1) begin : g_output_pack
            assign out_data_o[(feature_idx * 8) +: 8] = read_bank_r
                ? bank_1[read_row_ptr_r][feature_idx]
                : bank_0[read_row_ptr_r][feature_idx];
        end
    endgenerate

    integer token_idx;
    always @(posedge clk_i) begin
        if (!rst_ni) begin
            bank_0_full_r   <= 1'b0;
            bank_1_full_r   <= 1'b0;
            bank_0_tag_r    <= 5'd0;
            bank_1_tag_r    <= 5'd0;
            write_bank_r    <= 1'b0;
            read_bank_r     <= 1'b0;
            write_col_ptr_r <= 4'd0;
            read_row_ptr_r  <= 4'd0;
        end else begin
            if (input_fire_w) begin
                if (write_bank_r == 1'b0) begin
                    for (token_idx = 0; token_idx < 16;
                         token_idx = token_idx + 1) begin
                        bank_0[token_idx][in_col_i]
                            <= in_data_i[(token_idx * 8) +: 8];
                    end

                    if (write_col_ptr_r == 4'd0)
                        bank_0_tag_r <= in_tag_i;

                    if (write_col_ptr_r == 4'd15) begin
                        bank_0_full_r   <= 1'b1;
                        write_bank_r    <= 1'b1;
                        write_col_ptr_r <= 4'd0;
                    end else begin
                        write_col_ptr_r <= write_col_ptr_r + 4'd1;
                    end
                end else begin
                    for (token_idx = 0; token_idx < 16;
                         token_idx = token_idx + 1) begin
                        bank_1[token_idx][in_col_i]
                            <= in_data_i[(token_idx * 8) +: 8];
                    end

                    if (write_col_ptr_r == 4'd0)
                        bank_1_tag_r <= in_tag_i;

                    if (write_col_ptr_r == 4'd15) begin
                        bank_1_full_r   <= 1'b1;
                        write_bank_r    <= 1'b0;
                        write_col_ptr_r <= 4'd0;
                    end else begin
                        write_col_ptr_r <= write_col_ptr_r + 4'd1;
                    end
                end
            end

            if (output_fire_w) begin
                if (read_row_ptr_r == 4'd15) begin
                    read_row_ptr_r <= 4'd0;
                    read_bank_r    <= ~read_bank_r;

                    if (read_bank_r == 1'b0)
                        bank_0_full_r <= 1'b0;
                    else
                        bank_1_full_r <= 1'b0;
                end else begin
                    read_row_ptr_r <= read_row_ptr_r + 4'd1;
                end
            end
        end
    end

endmodule
