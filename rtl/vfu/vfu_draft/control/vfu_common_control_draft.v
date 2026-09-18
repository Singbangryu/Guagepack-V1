`timescale 1ns/1ps
`include "vfu_external_ops.vh"
`include "vfu_internal_op_defs.vh"

// Draft command dispatcher only. Datapath, FIFO, commit and row-state storage
// are external. In particular, returning to idle never clears Softmax R.
//
// SETUP is one cycle in this draft. The wrapper must buffer input until the
// selected child is active; SETUP does not repair upstream result outflow.
module vfu_common_control_draft (
    input  wire               clk_i,
    input  wire               rst_ni,       // Synchronous, active-low; idle reset.
    input  wire               start_i,      // One-cycle request while idle only.
    input  wire [2:0]         op_i,
    input  wire [9:0]         m_i,          // Padded token count: 16, 32, 48, 64.
    input  wire [9:0]         n_i,          // Features per command: 1..512.
    input  wire [6:0]         seq_len_i,    // Original sequence length: 2..64.
    input  wire               layer_i,
    input  wire               num_ln_i,
    input  wire signed [17:0] req_mult_i,
    input  wire [5:0]         req_shamt_i,
    input  wire               transpose_i,
    input  wire [2:0]         write_sel_i,
    input  wire [15:0]        write_base_i,
    input  wire [15:0]        res_base_i,
    input  wire [15:0]        ln_base_i,

    input  wire               rq_gelu_done_i,
    input  wire               softmax_done_i,
    input  wire               ln_done_i,

    output wire               cmd_accept_o,
    output wire               setup_o,
    output wire               busy_o,
    output wire               done_o,
    output wire               rq_gelu_init_o,
    output wire               rq_gelu_active_o,
    output wire               softmax_init_o,
    output wire               softmax_active_o,
    output wire               ln_init_o,
    output wire               ln_active_o,

    output wire [2:0]         op_o,
    output wire [9:0]         m_o,
    output wire [9:0]         n_o,
    output wire [6:0]         seq_len_o,
    output wire               layer_o,
    output wire               num_ln_o,
    output wire signed [17:0] req_mult_o,
    output wire [5:0]         req_shamt_o,
    output wire               transpose_o,
    output wire [2:0]         write_sel_o,
    output wire [15:0]        write_base_o,
    output wire [15:0]        res_base_o,
    output wire [15:0]        ln_base_o,
    output wire [2:0]         mt_o,
    output wire [11:0]        total_beats_o,
    // Static mapping for the RQ/GELU path only; ignore for other commands.
    // Softmax and LN controllers own their phase-dependent CORE micro-op.
    output wire [3:0]         rq_gelu_core_op_o
);

    localparam [1:0] COMMON_IDLE = 2'd0;
    localparam [1:0] SETUP       = 2'd1;
    localparam [1:0] CORE_RUN    = 2'd2;

    reg [1:0] state_r;
    reg [1:0] next_state;
    reg [2:0] op_r;
    reg [9:0] m_r, n_r;
    reg [6:0] seq_len_r;
    reg layer_r, num_ln_r;
    reg signed [17:0] req_mult_r;
    reg [5:0] req_shamt_r;
    reg transpose_r;
    reg [2:0] write_sel_r;
    reg [15:0] write_base_r, res_base_r, ln_base_r;

    wire op_legal;
    wire select_rq_gelu;
    wire select_softmax;
    wire select_ln;
    wire selected_done;

    assign op_legal = (op_i == `VFU_RQ) || (op_i == `VFU_GELU) ||
                      (op_i == `VFU_SM) || (op_i == `VFU_SM_NL) ||
                      (op_i == `VFU_LN);
    assign cmd_accept_o = rst_ni && (state_r == COMMON_IDLE) && start_i && op_legal;
    assign setup_o = (state_r == SETUP);
    assign busy_o = (state_r != COMMON_IDLE);

    assign select_rq_gelu = (op_r == `VFU_RQ) || (op_r == `VFU_GELU);
    assign select_softmax = (op_r == `VFU_SM) || (op_r == `VFU_SM_NL);
    assign select_ln = (op_r == `VFU_LN);
    assign rq_gelu_init_o = setup_o && select_rq_gelu;
    assign softmax_init_o = setup_o && select_softmax;
    assign ln_init_o = setup_o && select_ln;
    assign rq_gelu_active_o = (state_r == CORE_RUN) && select_rq_gelu;
    assign softmax_active_o = (state_r == CORE_RUN) && select_softmax;
    assign ln_active_o = (state_r == CORE_RUN) && select_ln;

    assign selected_done = (select_rq_gelu && rq_gelu_done_i) ||
                           (select_softmax && softmax_done_i) ||
                           (select_ln && ln_done_i);
    // Forward the selected child's completion in CORE_RUN. No extra DONE
    // register/state: descriptor and active remain live through this cycle.
    assign done_o = (state_r == CORE_RUN) && selected_done;

    always @(*) begin
        next_state = state_r;
        case (state_r)
            COMMON_IDLE: if (cmd_accept_o) next_state = SETUP;
            SETUP:       next_state = CORE_RUN;
            CORE_RUN:    if (done_o) next_state = COMMON_IDLE;
            default:     next_state = COMMON_IDLE;
        endcase
    end

    always @(posedge clk_i) begin
        if (!rst_ni) begin
            state_r <= COMMON_IDLE;
            op_r <= `VFU_RQ;
            m_r <= 10'd0;
            n_r <= 10'd0;
            seq_len_r <= 7'd0;
            layer_r <= 1'b0;
            num_ln_r <= 1'b0;
            req_mult_r <= 18'sd0;
            req_shamt_r <= 6'd0;
            transpose_r <= 1'b0;
            write_sel_r <= 3'd0;
            write_base_r <= 16'd0;
            res_base_r <= 16'd0;
            ln_base_r <= 16'd0;
        end else begin
            state_r <= next_state;
            if (cmd_accept_o) begin
                op_r <= op_i;
                m_r <= m_i;
                n_r <= n_i;
                seq_len_r <= seq_len_i;
                layer_r <= layer_i;
                num_ln_r <= num_ln_i;
                req_mult_r <= req_mult_i;
                req_shamt_r <= req_shamt_i;
                transpose_r <= transpose_i;
                write_sel_r <= write_sel_i;
                write_base_r <= write_base_i;
                res_base_r <= res_base_i;
                ln_base_r <= ln_base_i;
            end
        end
    end

    assign op_o = op_r;
    assign m_o = m_r;
    assign n_o = n_r;
    assign seq_len_o = seq_len_r;
    assign layer_o = layer_r;
    assign num_ln_o = num_ln_r;
    assign req_mult_o = req_mult_r;
    assign req_shamt_o = req_shamt_r;
    assign transpose_o = transpose_r;
    assign write_sel_o = write_sel_r;
    assign write_base_o = write_base_r;
    assign res_base_o = res_base_r;
    assign ln_base_o = ln_base_r;

    // All legal M values are multiples of 16. Keep the count wide enough for
    // MT=4, N=512 -> 2048 beats (distinct from a zero-based beat index).
    assign mt_o = m_r[6:4];
    assign total_beats_o = {9'd0, mt_o} * {2'd0, n_r};
    assign rq_gelu_core_op_o = (op_r == `VFU_GELU) ? `VFU_OP_GELU : `VFU_OP_RQ;

`ifndef SYNTHESIS
    // Illegal stimulus checks, not a hardware recovery or abort protocol.
    always @(posedge clk_i) begin
        if (rst_ni && start_i) begin
            if (state_r != COMMON_IDLE)
                $fatal(1, "common draft: start while busy");
            if (!op_legal)
                $fatal(1, "common draft: reserved external opcode");
            if ((m_i < 10'd16) || (m_i > 10'd64) || (m_i[3:0] != 4'd0))
                $fatal(1, "common draft: M must be 16/32/48/64");
            if ((n_i == 10'd0) || (n_i > 10'd512))
                $fatal(1, "common draft: N must be 1..512");
            if ((seq_len_i < 7'd2) || (seq_len_i > 7'd64))
                $fatal(1, "common draft: seq_len must be 2..64");
        end
    end
`endif
endmodule
