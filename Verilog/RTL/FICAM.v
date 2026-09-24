module FICAM #(
    parameter integer PPN_LENGTH    = 16,
    parameter integer BANK_LENGTH   = 3,
    parameter integer RG_LENGTH     = 3,
    parameter integer FW_LENGTH     = 3,
    parameter integer FP_LENGTH     = 3,
    parameter integer CW_LENGTH     = 3,
    parameter integer FICAM_ENTRIES = 16,
    parameter integer FI_PER_ENTRY  = 2,
    parameter integer FI_SLOT_WIDTH = 1 + FW_LENGTH + FP_LENGTH,
    parameter integer FICAM_LENGTH  = 1 + PPN_LENGTH + BANK_LENGTH + RG_LENGTH
                                      + FI_PER_ENTRY*FI_SLOT_WIDTH,
    parameter integer INDEX_WIDTH   = (FICAM_ENTRIES <= 2) ? 1 : $clog2(FICAM_ENTRIES),
    parameter integer COUNT_WIDTH   = (FICAM_ENTRIES <= 1) ? 1 : $clog2(FICAM_ENTRIES + 1)
)(
    input  wire                                  clk,
    input  wire                                  rst_n,

    // -------------------------- LOAD phase --------------------------
    input  wire                                  Group_Is_Exp,
    input  wire [FICAM_LENGTH-1:0]               FICAM_Entry,
    input  wire                                  Load_Lock,
    output wire                                  FICAM_Full,
    output wire [COUNT_WIDTH-1:0]                Entry_Count_Out,

    // -------------------------- EVAL direct read --------------------
    input  wire                                  Eval_Read_Enable,
    input  wire [INDEX_WIDTH-1:0]                Eval_Read_Index,
    output reg                                   Eval_Read_Valid,
    output reg  [BANK_LENGTH-1:0]                Eval_Read_Bank,
    output reg  [FI_PER_ENTRY-1:0]               Eval_Read_FI_Valid,
    output reg  [FI_PER_ENTRY*FP_LENGTH-1:0]     Eval_Read_FI_FP,
    output reg                                   Eval_Read_Is_Exp,

    // -------------------------- CU result write-back ----------------
    input  wire                                  Config_Write_Enable,
    input  wire [INDEX_WIDTH-1:0]                Config_Index,
    input  wire [1:0]                            Config_Remap_Mode,
    input  wire [CW_LENGTH-1:0]                  Config_Remap_Value,

    // -------------------------- Normal lookup -----------------------
    input  wire                                  Search_Enable,
    input  wire [PPN_LENGTH-1:0]                 Search_PPN,
    input  wire [BANK_LENGTH-1:0]                Search_Bank,
    input  wire [RG_LENGTH-1:0]                  Search_RG,
    output reg                                   Search_Match,
    output reg  [1:0]                            Remap_Mode_Out,
    output reg  [CW_LENGTH-1:0]                  Remap_Value_Out
);

localparam [1:0] REMAP_NONE = 2'b00;

reg                                  Valid_Mem       [0:FICAM_ENTRIES-1];
reg [PPN_LENGTH-1:0]                 PPN_Mem         [0:FICAM_ENTRIES-1];
reg [BANK_LENGTH-1:0]                Bank_Mem        [0:FICAM_ENTRIES-1];
reg [RG_LENGTH-1:0]                  RG_Mem          [0:FICAM_ENTRIES-1];
reg                                  Exp_Mem         [0:FICAM_ENTRIES-1];
reg [FI_PER_ENTRY-1:0]               FI_Valid_Mem    [0:FICAM_ENTRIES-1];
reg [FI_PER_ENTRY*FW_LENGTH-1:0]     FI_FW_Mem       [0:FICAM_ENTRIES-1];
reg [FI_PER_ENTRY*FP_LENGTH-1:0]     FI_FP_Mem       [0:FICAM_ENTRIES-1];
reg [1:0]                            Mode_Mem        [0:FICAM_ENTRIES-1];
reg [CW_LENGTH-1:0]                  Remap_Value_Mem [0:FICAM_ENTRIES-1];

reg [COUNT_WIDTH-1:0] Entry_Count;
assign Entry_Count_Out = Entry_Count;
assign FICAM_Full = (Entry_Count == FICAM_ENTRIES);

localparam integer FI0_FP_LSB = 0;
localparam integer FI0_FW_LSB = FI0_FP_LSB + FP_LENGTH;
localparam integer FI0_V_BIT  = FI0_FW_LSB + FW_LENGTH;
localparam integer FI1_FP_LSB = FI_SLOT_WIDTH;
localparam integer FI1_FW_LSB = FI1_FP_LSB + FP_LENGTH;
localparam integer FI1_V_BIT  = FI1_FW_LSB + FW_LENGTH;
localparam integer RG_LSB     = FI_PER_ENTRY * FI_SLOT_WIDTH;
localparam integer BANK_LSB   = RG_LSB + RG_LENGTH;
localparam integer PPN_LSB    = BANK_LSB + BANK_LENGTH;
localparam integer VALID_BIT  = PPN_LSB + PPN_LENGTH;

wire                                  Incoming_Valid;
wire [PPN_LENGTH-1:0]                 Incoming_PPN;
wire [BANK_LENGTH-1:0]                Incoming_Bank;
wire [RG_LENGTH-1:0]                  Incoming_RG;
wire [FI_PER_ENTRY-1:0]               Incoming_FI_Valid;
wire [FI_PER_ENTRY*FW_LENGTH-1:0]     Incoming_FI_FW;
wire [FI_PER_ENTRY*FP_LENGTH-1:0]     Incoming_FI_FP;

assign Incoming_Valid = FICAM_Entry[VALID_BIT];
assign Incoming_PPN   = FICAM_Entry[PPN_LSB  +: PPN_LENGTH];
assign Incoming_Bank  = FICAM_Entry[BANK_LSB +: BANK_LENGTH];
assign Incoming_RG    = FICAM_Entry[RG_LSB   +: RG_LENGTH];

assign Incoming_FI_Valid[0] = FICAM_Entry[FI0_V_BIT];
assign Incoming_FI_FW[0*FW_LENGTH +: FW_LENGTH] =
    FICAM_Entry[FI0_FW_LSB +: FW_LENGTH];
assign Incoming_FI_FP[0*FP_LENGTH +: FP_LENGTH] =
    FICAM_Entry[FI0_FP_LSB +: FP_LENGTH];

assign Incoming_FI_Valid[1] = FICAM_Entry[FI1_V_BIT];
assign Incoming_FI_FW[1*FW_LENGTH +: FW_LENGTH] =
    FICAM_Entry[FI1_FW_LSB +: FW_LENGTH];
assign Incoming_FI_FP[1*FP_LENGTH +: FP_LENGTH] =
    FICAM_Entry[FI1_FP_LSB +: FP_LENGTH];

`ifndef SYNTHESIS
initial begin
    if (FI_PER_ENTRY != 2)
        $error("FICAM packing currently requires FI_PER_ENTRY=2");
end
`endif

integer i;
integer Matched_Index;
integer Empty_Index;
reg     Matched_Found;
reg     Empty_Found;

always @(posedge clk) begin
    if (!rst_n) begin
        Entry_Count <= {COUNT_WIDTH{1'b0}};
        for (i = 0; i < FICAM_ENTRIES; i = i + 1) begin
            Valid_Mem[i]       <= 1'b0;
            PPN_Mem[i]         <= {PPN_LENGTH{1'b0}};
            Bank_Mem[i]        <= {BANK_LENGTH{1'b0}};
            RG_Mem[i]          <= {RG_LENGTH{1'b0}};
            Exp_Mem[i]         <= 1'b0;
            FI_Valid_Mem[i]    <= {FI_PER_ENTRY{1'b0}};
            FI_FW_Mem[i]       <= {(FI_PER_ENTRY*FW_LENGTH){1'b0}};
            FI_FP_Mem[i]       <= {(FI_PER_ENTRY*FP_LENGTH){1'b0}};
            Mode_Mem[i]        <= REMAP_NONE;
            Remap_Value_Mem[i] <= {CW_LENGTH{1'b0}};
        end
    end
    else begin
        // CU repair-result write-back is allowed after loading is locked,
        // because it changes only remap information, not FI payload.
        if (Config_Write_Enable) begin
            Mode_Mem[Config_Index]        <= Config_Remap_Mode;
            Remap_Value_Mem[Config_Index] <= Config_Remap_Value;
        end

        if (Incoming_Valid && !FICAM_Full && !Load_Lock) begin
            Matched_Found = 1'b0;
            Empty_Found   = 1'b0;
            Matched_Index = 0;
            Empty_Index   = 0;

            for (i = 0; i < FICAM_ENTRIES; i = i + 1) begin
                if (!Matched_Found && Valid_Mem[i] &&
                    (PPN_Mem[i]  == Incoming_PPN) &&
                    (Bank_Mem[i] == Incoming_Bank) &&
                    (RG_Mem[i]   == Incoming_RG)) begin
                    Matched_Found = 1'b1;
                    Matched_Index = i;
                end

                if (!Empty_Found && !Valid_Mem[i]) begin
                    Empty_Found = 1'b1;
                    Empty_Index = i;
                end
            end

            if (Matched_Found || Empty_Found) begin
                if (!Matched_Found) begin
                    Matched_Index = Empty_Index;
                    Entry_Count <= Entry_Count + {{(COUNT_WIDTH-1){1'b0}}, 1'b1};
                end

                Valid_Mem[Matched_Index]    <= 1'b1;
                PPN_Mem[Matched_Index]      <= Incoming_PPN;
                Bank_Mem[Matched_Index]     <= Incoming_Bank;
                RG_Mem[Matched_Index]       <= Incoming_RG;
                Exp_Mem[Matched_Index]      <= Group_Is_Exp;
                FI_Valid_Mem[Matched_Index] <= Incoming_FI_Valid;
                FI_FW_Mem[Matched_Index]    <= Incoming_FI_FW;
                FI_FP_Mem[Matched_Index]    <= Incoming_FI_FP;
                Mode_Mem[Matched_Index]        <= REMAP_NONE;
                Remap_Value_Mem[Matched_Index] <= {CW_LENGTH{1'b0}};
            end
        end
    end
end

// Direct physical-index read for the evaluation pipeline.
// With 16 entries this synthesizes to a small mux; there is no sequential CAM
// search and no scan from Entry 0 looking for a requested PPN/RG.
always @(*) begin
    Eval_Read_Valid    = 1'b0;
    Eval_Read_Bank     = {BANK_LENGTH{1'b0}};
    Eval_Read_FI_Valid = {FI_PER_ENTRY{1'b0}};
    Eval_Read_FI_FP    = {(FI_PER_ENTRY*FP_LENGTH){1'b0}};
    Eval_Read_Is_Exp   = 1'b0;

    if (Eval_Read_Enable && (Eval_Read_Index < FICAM_ENTRIES) &&
        Valid_Mem[Eval_Read_Index]) begin
        Eval_Read_Valid    = 1'b1;
        Eval_Read_Bank     = Bank_Mem[Eval_Read_Index];
        Eval_Read_FI_Valid = FI_Valid_Mem[Eval_Read_Index];
        Eval_Read_FI_FP    = FI_FP_Mem[Eval_Read_Index];
        Eval_Read_Is_Exp   = Exp_Mem[Eval_Read_Index];
    end
end

// Associative normal-operation lookup. All 16 tags are compared in parallel.
integer s;
always @(*) begin
    Search_Match    = 1'b0;
    Remap_Mode_Out  = REMAP_NONE;
    Remap_Value_Out = {CW_LENGTH{1'b0}};

    if (Search_Enable) begin
        for (s = 0; s < FICAM_ENTRIES; s = s + 1) begin
            if (!Search_Match && Valid_Mem[s] &&
                (PPN_Mem[s]  == Search_PPN) &&
                (Bank_Mem[s] == Search_Bank) &&
                (RG_Mem[s]   == Search_RG)) begin
                Search_Match    = 1'b1;
                Remap_Mode_Out  = Mode_Mem[s];
                Remap_Value_Out = Remap_Value_Mem[s];
            end
        end
    end
end

endmodule
