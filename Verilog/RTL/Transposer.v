module Transposer #(
    parameter integer WORD_COUNT   = 8,
    parameter integer WW_LENGTH    = 8,
    parameter integer TWW_LENGTH   = 8,
    parameter integer COUNT_LENGTH = 3
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  Start,

    // Direction 1: WW -> TWW. Direction 0: TWW -> WW.
    input  wire                  Direction,
    input  wire                  Data_In_Valid,
    output wire                  Input_Ready,

    input  wire [WW_LENGTH-1:0]  WW_Data_In,
    input  wire [TWW_LENGTH-1:0] TWW_Data_In,
    output reg  [WW_LENGTH-1:0]  WW_Data_Out,
    output reg  [TWW_LENGTH-1:0] TWW_Data_Out,

    output wire                  Data_Out_Valid,
    output wire                  Busy,
    output reg                   Done
);

localparam [1:0] IDLE        = 2'd0;
localparam [1:0] LOAD        = 2'd1;
localparam [1:0] OUTPUT_DATA = 2'd2;
localparam [1:0] FINISH      = 2'd3;

reg [1:0]                State;
reg [COUNT_LENGTH-1:0]   Load_Counter;
reg [COUNT_LENGTH-1:0]   Output_Counter;
reg                      Direction_Latched;
reg [WW_LENGTH-1:0]      Vector_Mem [0:WORD_COUNT-1];
integer                  Output_Bit_Index;
integer                  Reset_Index;

assign Input_Ready    = (State == LOAD);
assign Data_Out_Valid = (State == OUTPUT_DATA);
assign Busy           = (State != IDLE);

always @(*) begin
    WW_Data_Out  = {WW_LENGTH{1'b0}};
    TWW_Data_Out = {TWW_LENGTH{1'b0}};

    if (State == OUTPUT_DATA) begin
        if (Direction_Latched) begin
            for (Output_Bit_Index = 0; Output_Bit_Index < TWW_LENGTH;
                 Output_Bit_Index = Output_Bit_Index + 1)
                TWW_Data_Out[Output_Bit_Index] =
                    Vector_Mem[Output_Bit_Index][Output_Counter];
        end
        else begin
            for (Output_Bit_Index = 0; Output_Bit_Index < WW_LENGTH;
                 Output_Bit_Index = Output_Bit_Index + 1)
                WW_Data_Out[Output_Bit_Index] =
                    Vector_Mem[Output_Bit_Index][Output_Counter];
        end
    end
end

always @(posedge clk) begin
    if (!rst_n) begin
        State              <= IDLE;
        Load_Counter       <= {COUNT_LENGTH{1'b0}};
        Output_Counter     <= {COUNT_LENGTH{1'b0}};
        Direction_Latched  <= 1'b0;
        Done               <= 1'b0;
        for (Reset_Index = 0; Reset_Index < WORD_COUNT; Reset_Index = Reset_Index + 1)
            Vector_Mem[Reset_Index] <= {WW_LENGTH{1'b0}};
    end
    else begin
        Done <= 1'b0;
        case (State)
            IDLE: begin
                Load_Counter   <= {COUNT_LENGTH{1'b0}};
                Output_Counter <= {COUNT_LENGTH{1'b0}};
                if (Start) begin
                    Direction_Latched <= Direction;
                    State             <= LOAD;
                end
            end

            LOAD: begin
                if (Data_In_Valid) begin
                    Vector_Mem[Load_Counter] <= Direction_Latched
                        ? WW_Data_In : TWW_Data_In;

                    if (Load_Counter == WORD_COUNT-1) begin
                        Load_Counter   <= {COUNT_LENGTH{1'b0}};
                        Output_Counter <= {COUNT_LENGTH{1'b0}};
                        State          <= OUTPUT_DATA;
                    end
                    else begin
                        Load_Counter <= Load_Counter + 1'b1;
                    end
                end
            end

            OUTPUT_DATA: begin
                if (Output_Counter == WORD_COUNT-1) begin
                    Output_Counter <= {COUNT_LENGTH{1'b0}};
                    State          <= FINISH;
                end
                else begin
                    Output_Counter <= Output_Counter + 1'b1;
                end
            end

            FINISH: begin
                Done  <= 1'b1;
                State <= IDLE;
            end

            default: State <= IDLE;
        endcase
    end
end

endmodule
