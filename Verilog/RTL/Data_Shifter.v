module Data_Shifter #(
    parameter integer SM_LENGTH   = 3,
    parameter integer DATA_LENGTH = 8
)(
    input  wire [SM_LENGTH-1:0]   Shift_Mode,
    input  wire [DATA_LENGTH-1:0] Switched_Data_In,
    input  wire [DATA_LENGTH-1:0] Shifted_Data_In,
    output reg  [DATA_LENGTH-1:0] Switched_Data_Out,
    output reg  [DATA_LENGTH-1:0] Shifted_Data_Out
);

always @(*) begin
    if (Shift_Mode == {SM_LENGTH{1'b0}}) begin
        Shifted_Data_Out  = Switched_Data_In;
        Switched_Data_Out = Shifted_Data_In;
    end
    else begin
        // Write path: rotate right.
        Shifted_Data_Out = (Switched_Data_In >> Shift_Mode) |
                           (Switched_Data_In << (DATA_LENGTH - Shift_Mode));
        // Read path: inverse rotation (rotate left).
        Switched_Data_Out = (Shifted_Data_In << Shift_Mode) |
                            (Shifted_Data_In >> (DATA_LENGTH - Shift_Mode));
    end
end

endmodule
