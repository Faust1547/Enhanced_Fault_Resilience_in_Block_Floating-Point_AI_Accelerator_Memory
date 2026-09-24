module ES_Score_Engine #(
    parameter integer FI_PER_ENTRY = 2,
    parameter integer FP_LENGTH    = 3,
    parameter integer CW_LENGTH    = 3,
    parameter integer ES_WIDTH     = 16,
    parameter integer CANDIDATES   = 8
)(
    input  wire [FI_PER_ENTRY-1:0]             FI_Valid,
    input  wire [FI_PER_ENTRY*FP_LENGTH-1:0]   FI_FP,
    input  wire                                Group_Is_Exp,
    output wire [ES_WIDTH-1:0]                 Base_ES,
    output wire [CANDIDATES*ES_WIDTH-1:0]      Intra_ES_Packed,
    output wire [CANDIDATES*ES_WIDTH-1:0]      Inter_ES_Packed
);
wire [FP_LENGTH-1:0] FP0;
wire [FP_LENGTH-1:0] FP1;
assign FP0 = FI_FP[0*FP_LENGTH +: FP_LENGTH];
assign FP1 = FI_FP[1*FP_LENGTH +: FP_LENGTH];

wire [ES_WIDTH-1:0] Base_0;
wire [ES_WIDTH-1:0] Base_1;
assign Base_0 = FI_Valid[0] ? Weight(Group_Is_Exp, FP0) : {ES_WIDTH{1'b0}};
assign Base_1 = FI_Valid[1] ? Weight(Group_Is_Exp, FP1) : {ES_WIDTH{1'b0}};
assign Base_ES = Base_0 + Base_1;

genvar c;
generate
    for (c = 0; c < CANDIDATES; c = c + 1) begin : g_candidate
        localparam [FP_LENGTH-1:0] CANDIDATE_VALUE = c;
        wire [FP_LENGTH-1:0] Intra_FP0;
        wire [FP_LENGTH-1:0] Intra_FP1;
        wire [FP_LENGTH-1:0] Inter_FP0;
        wire [FP_LENGTH-1:0] Inter_FP1;
        wire [ES_WIDTH-1:0] Intra_0;
        wire [ES_WIDTH-1:0] Intra_1;
        wire [ES_WIDTH-1:0] Inter_0;
        wire [ES_WIDTH-1:0] Inter_1;

        // Intra: the shared CW remaps the 3-bit logical position by XOR.
        assign Intra_FP0 = FP0 ^ CANDIDATE_VALUE;
        assign Intra_FP1 = FP1 ^ CANDIDATE_VALUE;

        // Inter: one shift value is shared by the FI list.  Three-bit
        // overflow naturally performs modulo 8.
        assign Inter_FP0 = FP0 + CANDIDATE_VALUE;
        assign Inter_FP1 = FP1 + CANDIDATE_VALUE;

        assign Intra_0 = FI_Valid[0]
            ? Weight(Group_Is_Exp, Intra_FP0) : {ES_WIDTH{1'b0}};
        assign Intra_1 = FI_Valid[1]
            ? Weight(Group_Is_Exp, Intra_FP1) : {ES_WIDTH{1'b0}};
        assign Inter_0 = FI_Valid[0]
            ? Weight(Group_Is_Exp, Inter_FP0) : {ES_WIDTH{1'b0}};
        assign Inter_1 = FI_Valid[1]
            ? Weight(Group_Is_Exp, Inter_FP1) : {ES_WIDTH{1'b0}};

        assign Intra_ES_Packed[c*ES_WIDTH +: ES_WIDTH] = Intra_0 + Intra_1;
        assign Inter_ES_Packed[c*ES_WIDTH +: ES_WIDTH] = Inter_0 + Inter_1;
    end
endgenerate

// Power-of-two weights: synthesized as shifts/wiring, never multipliers.
function [ES_WIDTH-1:0] Weight;
    input                  Is_Exp;
    input [FP_LENGTH-1:0]  Logical_Bit;
    reg   [ES_WIDTH-1:0]   One;
    begin
        One = {{(ES_WIDTH-1){1'b0}}, 1'b1};
        if (Is_Exp) begin
            case (Logical_Bit)
                3'd7: Weight = One << 5; // 32
                3'd2,
                3'd1: Weight = One << 4; // 16
                default: Weight = One << 3; // 8
            endcase
        end
        else begin
            case (Logical_Bit)
                3'd7: Weight = One << 5; // 32
                3'd6: Weight = One << 4; // 16
                3'd5: Weight = One << 3; // 8
                3'd4: Weight = One << 2; // 4
                default: Weight = One << 1; // 2
            endcase
        end
    end
endfunction

endmodule
