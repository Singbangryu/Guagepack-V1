
module vfu_rq_gelu_control 
(   

    input wire clk_i,
    input wire rstn_i,

    input wire init_i,
    input wire active_i,

    input wire beat_valid_i,

    input wire commit_done_i,



    input wire [11:0] total_beats_i,

    output wire      rq_gelu_done_o,
    output wire      stream_en_o,
    output wire      last_beat_o

);
    
    localparam STREAM = 1'b0;
    localparam DRAIN  = 1'b1;

    reg         curr_state;
    reg         next_state;         
    reg  [11:0] beat_counter_r;


    wire beat_fire;
    wire last_beat;
    wire stream_en;
    assign stream_en = active_i && (curr_state == STREAM);
    assign beat_fire = stream_en && beat_valid_i;
    assign last_beat = beat_fire && (beat_counter_r == total_beats_i - 12'd1);


    
    always @(posedge clk_i) begin
        if(~rstn_i | init_i) begin
            curr_state <= STREAM;
        end

        else begin
            curr_state <= next_state;
        end
    end

    always @(posedge clk_i) begin
        if(~rstn_i | init_i) begin
            beat_counter_r <= 12'd0;
        end

        else begin
            if(beat_fire) 
                beat_counter_r <= beat_counter_r + 12'd1;           

        end

    end


    always @(*) begin
       next_state = curr_state;
        
        case(curr_state) 

            STREAM : begin
                next_state = (last_beat) ? DRAIN : STREAM;

            end

            DRAIN : begin
            
                next_state = DRAIN;
            end
            
            default : 
                next_state = STREAM;
        endcase

        

    end
    assign rq_gelu_done_o = active_i && commit_done_i && (curr_state == DRAIN);
    assign last_beat_o = last_beat;
    assign stream_en_o = stream_en;








endmodule