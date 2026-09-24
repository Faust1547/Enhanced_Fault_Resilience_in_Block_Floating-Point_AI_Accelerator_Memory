// Pipeline:
//   S1: direct FICAM record fetch/capture
//   S2: Base + 8 Intra + 8 Inter candidate ES generation
//   S3A: INTRA pairwise minimum (8 -> 4), then register
//   S3B: INTRA final minimum (4 -> 2 -> 1), retain 8 Inter candidates
//   S4 : page Base/Intra accumulation + bank/shift Inter accumulation
//   S5 : per-Bank INTER 8-to-1 balanced minimum, then register results
//   S6 : balanced 8-Bank ES sum, then register Page INTER ES
//   S7 : final Base / Inter / Intra page-mode compare
//   S8 : sequential FICAM remap-information write-back
module Control_Unit #(
    parameter integer FP_LENGTH      = 3,
    parameter integer CW_LENGTH      = 3,
    parameter integer ES_WIDTH       = 16,
    parameter integer FI_PER_ENTRY   = 2,
    parameter integer FICAM_ENTRIES  = 16,
    parameter integer BANKS          = 8,
    parameter integer CANDIDATES     = 8,
    parameter integer INDEX_WIDTH    = (FICAM_ENTRIES <= 2) ? 1 : $clog2(FICAM_ENTRIES),
    parameter integer COUNT_WIDTH    = (FICAM_ENTRIES <= 1) ? 1 : $clog2(FICAM_ENTRIES + 1),
    parameter integer BANK_WIDTH     = (BANKS <= 2) ? 1 : $clog2(BANKS),
    parameter integer PAGE_ES_WIDTH  = ES_WIDTH + COUNT_WIDTH
)(
    input  wire                                  clk,
    input  wire                                  rst_n,

    input  wire                                  Eval_Start,
    input  wire [COUNT_WIDTH-1:0]                Eval_Entry_Count,

    output wire                                  Eval_Read_Enable,
    output wire [INDEX_WIDTH-1:0]                Eval_Read_Index,
    input  wire                                  Eval_Read_Valid,
    input  wire [BANK_WIDTH-1:0]                 Eval_Read_Bank,
    input  wire [FI_PER_ENTRY-1:0]               Eval_Read_FI_Valid,
    input  wire [FI_PER_ENTRY*FP_LENGTH-1:0]     Eval_Read_FI_FP,
    input  wire                                  Eval_Read_Is_Exp,

    output reg                                   Config_Write_Enable,
    output reg  [INDEX_WIDTH-1:0]                Config_Index,
    output reg  [1:0]                            Config_Remap_Mode,
    output reg  [CW_LENGTH-1:0]                  Config_Remap_Value,

    output reg  [1:0]                            Remap_Mode,
    output wire                                  Busy,
    output reg                                   Eval_Done
);

localparam [1:0] REMAP_NONE       = 2'b00;
localparam [1:0] REMAP_INTER_BANK = 2'b01;
localparam [1:0] REMAP_INTRA_BANK = 2'b10;
localparam integer INTER_ACCUMS   = BANKS * CANDIDATES;

// -------------------------------------------------------------------------
// Fetch controller feeding S1.
// -------------------------------------------------------------------------
reg                         Fetch_Active;
reg [INDEX_WIDTH-1:0]       Fetch_Index;
reg [COUNT_WIDTH-1:0]       Eval_Count_Latched;
reg                         Page_Closing;

assign Eval_Read_Enable = Fetch_Active;
assign Eval_Read_Index  = Fetch_Index;

wire [COUNT_WIDTH-1:0] Fetch_Position_Next =
    {{(COUNT_WIDTH-INDEX_WIDTH){1'b0}}, Fetch_Index} + 1'b1;

// -------------------------------------------------------------------------
// S1 : capture one directly-indexed FICAM record.
// -------------------------------------------------------------------------
reg                                   S1_Valid;
reg [INDEX_WIDTH-1:0]                 S1_Index;
reg [BANK_WIDTH-1:0]                  S1_Bank;
reg [FI_PER_ENTRY-1:0]                S1_FI_Valid;
reg [FI_PER_ENTRY*FP_LENGTH-1:0]      S1_FI_FP;
reg                                   S1_Is_Exp;
reg                                   S1_Page_Last;

wire [ES_WIDTH-1:0]                   Base_ES_Wire;
wire [CANDIDATES*ES_WIDTH-1:0]        Intra_ES_Wire;
wire [CANDIDATES*ES_WIDTH-1:0]        Inter_ES_Wire;

ES_Score_Engine #(
    .FI_PER_ENTRY (FI_PER_ENTRY),
    .FP_LENGTH    (FP_LENGTH),
    .CW_LENGTH    (CW_LENGTH),
    .ES_WIDTH     (ES_WIDTH),
    .CANDIDATES   (CANDIDATES)
) u_es_score_engine (
    .FI_Valid        (S1_FI_Valid),
    .FI_FP           (S1_FI_FP),
    .Group_Is_Exp    (S1_Is_Exp),
    .Base_ES         (Base_ES_Wire),
    .Intra_ES_Packed (Intra_ES_Wire),
    .Inter_ES_Packed (Inter_ES_Wire)
);

// -------------------------------------------------------------------------
// S2 : register all candidate ES values.
// -------------------------------------------------------------------------
reg                                   S2_Valid;
reg [INDEX_WIDTH-1:0]                 S2_Index;
reg [BANK_WIDTH-1:0]                  S2_Bank;
reg [ES_WIDTH-1:0]                    S2_Base_ES;
reg [CANDIDATES*ES_WIDTH-1:0]         S2_Intra_ES;
reg [CANDIDATES*ES_WIDTH-1:0]         S2_Inter_ES;
reg                                   S2_Page_Last;

// -------------------------------------------------------------------------
// S3A : first INTRA minimum level (8 -> 4) only.
//       Registering these four pair winners cuts the former S2 -> S3
//       three-comparator path at the exact STA bottleneck.
// -------------------------------------------------------------------------
wire [ES_WIDTH-1:0] I0 = S2_Intra_ES[0*ES_WIDTH +: ES_WIDTH];
wire [ES_WIDTH-1:0] I1 = S2_Intra_ES[1*ES_WIDTH +: ES_WIDTH];
wire [ES_WIDTH-1:0] I2 = S2_Intra_ES[2*ES_WIDTH +: ES_WIDTH];
wire [ES_WIDTH-1:0] I3 = S2_Intra_ES[3*ES_WIDTH +: ES_WIDTH];
wire [ES_WIDTH-1:0] I4 = S2_Intra_ES[4*ES_WIDTH +: ES_WIDTH];
wire [ES_WIDTH-1:0] I5 = S2_Intra_ES[5*ES_WIDTH +: ES_WIDTH];
wire [ES_WIDTH-1:0] I6 = S2_Intra_ES[6*ES_WIDTH +: ES_WIDTH];
wire [ES_WIDTH-1:0] I7 = S2_Intra_ES[7*ES_WIDTH +: ES_WIDTH];

wire [ES_WIDTH-1:0] I01 = (I1 < I0) ? I1 : I0;
wire [ES_WIDTH-1:0] I23 = (I3 < I2) ? I3 : I2;
wire [ES_WIDTH-1:0] I45 = (I5 < I4) ? I5 : I4;
wire [ES_WIDTH-1:0] I67 = (I7 < I6) ? I7 : I6;
wire [CW_LENGTH-1:0] I01_IDX = (I1 < I0) ? 3'd1 : 3'd0;
wire [CW_LENGTH-1:0] I23_IDX = (I3 < I2) ? 3'd3 : 3'd2;
wire [CW_LENGTH-1:0] I45_IDX = (I5 < I4) ? 3'd5 : 3'd4;
wire [CW_LENGTH-1:0] I67_IDX = (I7 < I6) ? 3'd7 : 3'd6;

reg                                   S3A_Valid;
reg [INDEX_WIDTH-1:0]                 S3A_Index;
reg [BANK_WIDTH-1:0]                  S3A_Bank;
reg [ES_WIDTH-1:0]                    S3A_Base_ES;
reg [ES_WIDTH-1:0]                    S3A_I01_ES;
reg [ES_WIDTH-1:0]                    S3A_I23_ES;
reg [ES_WIDTH-1:0]                    S3A_I45_ES;
reg [ES_WIDTH-1:0]                    S3A_I67_ES;
reg [CW_LENGTH-1:0]                   S3A_I01_IDX;
reg [CW_LENGTH-1:0]                   S3A_I23_IDX;
reg [CW_LENGTH-1:0]                   S3A_I45_IDX;
reg [CW_LENGTH-1:0]                   S3A_I67_IDX;
reg [CANDIDATES*ES_WIDTH-1:0]         S3A_Inter_ES;
reg                                   S3A_Page_Last;

// -------------------------------------------------------------------------
// S3B : remaining INTRA minimum levels (4 -> 2 -> 1).
// -------------------------------------------------------------------------
wire [ES_WIDTH-1:0] I0123 =
    (S3A_I23_ES < S3A_I01_ES) ? S3A_I23_ES : S3A_I01_ES;
wire [ES_WIDTH-1:0] I4567 =
    (S3A_I67_ES < S3A_I45_ES) ? S3A_I67_ES : S3A_I45_ES;
wire [CW_LENGTH-1:0] I0123_IDX =
    (S3A_I23_ES < S3A_I01_ES) ? S3A_I23_IDX : S3A_I01_IDX;
wire [CW_LENGTH-1:0] I4567_IDX =
    (S3A_I67_ES < S3A_I45_ES) ? S3A_I67_IDX : S3A_I45_IDX;

wire [ES_WIDTH-1:0] Best_Intra_ES_Wire =
    (I4567 < I0123) ? I4567 : I0123;
wire [CW_LENGTH-1:0] Best_Intra_CW_Wire =
    (I4567 < I0123) ? I4567_IDX : I0123_IDX;

// Predecode Bank ID one stage before S4.  The decoder is therefore placed
// on the S3A -> S3 register path instead of the S3 -> S4 Inter-accumulator
// critical path.  S4 sees only registered one-hot enables.
wire [BANKS-1:0] S3_Bank_OneHot_Wire =
    ({{(BANKS-1){1'b0}}, 1'b1} << S3A_Bank);

reg                                   S3_Valid;
reg [INDEX_WIDTH-1:0]                 S3_Index;
reg [BANK_WIDTH-1:0]                  S3_Bank;
reg [BANKS-1:0]                       S3_Bank_OneHot;
reg [ES_WIDTH-1:0]                    S3_Base_ES;
reg [ES_WIDTH-1:0]                    S3_Best_Intra_ES;
reg [CW_LENGTH-1:0]                   S3_Best_CW;
reg [CANDIDATES*ES_WIDTH-1:0]         S3_Inter_ES;
reg                                   S3_Page_Last;

// -------------------------------------------------------------------------
// S4 : only accumulation and per-entry metadata buffering.
//      Bank selection for Inter accumulation is already registered one-hot,
//      so this stage contains no binary Bank decoder/dynamic array index.
//      No final minimum tree or page mode comparison is on this stage.
// -------------------------------------------------------------------------
reg [PAGE_ES_WIDTH-1:0] Page_Base_ES;
reg [PAGE_ES_WIDTH-1:0] Page_Intra_ES;
reg [PAGE_ES_WIDTH-1:0] Page_Inter_ES [0:INTER_ACCUMS-1];

reg [COUNT_WIDTH-1:0]   Page_Entry_Count;
reg [INDEX_WIDTH-1:0]   Page_Index_Buffer [0:FICAM_ENTRIES-1];
reg [BANK_WIDTH-1:0]    Page_Bank_Buffer  [0:FICAM_ENTRIES-1];
reg [CW_LENGTH-1:0]     Page_CW_Buffer    [0:FICAM_ENTRIES-1];

reg                     Finalize_Pending;

wire [PAGE_ES_WIDTH-1:0] S4_Base_Ext =
    {{(PAGE_ES_WIDTH-ES_WIDTH){1'b0}}, S3_Base_ES};
wire [PAGE_ES_WIDTH-1:0] S4_Intra_Ext =
    {{(PAGE_ES_WIDTH-ES_WIDTH){1'b0}}, S3_Best_Intra_ES};

// -------------------------------------------------------------------------
// S5 : per-Bank INTER balanced 8-to-1 minimum only.
//      The result for each Bank is registered before any Bank summation.
// -------------------------------------------------------------------------
reg [PAGE_ES_WIDTH-1:0] Inter_L1_ES    [0:BANKS*4-1];
reg [CW_LENGTH-1:0]     Inter_L1_Shift [0:BANKS*4-1];
reg [PAGE_ES_WIDTH-1:0] Inter_L2_ES    [0:BANKS*2-1];
reg [CW_LENGTH-1:0]     Inter_L2_Shift [0:BANKS*2-1];
reg [PAGE_ES_WIDTH-1:0] Bank_Best_ES_Comb    [0:BANKS-1];
reg [CW_LENGTH-1:0]     Bank_Best_Shift_Comb [0:BANKS-1];

integer bi;
integer pair_base;
always @(*) begin
    // Level 1: (0,1), (2,3), (4,5), (6,7)
    for (bi = 0; bi < BANKS; bi = bi + 1) begin
        pair_base = bi * CANDIDATES;

        if (Page_Inter_ES[pair_base+1] < Page_Inter_ES[pair_base+0]) begin
            Inter_L1_ES[bi*4+0]    = Page_Inter_ES[pair_base+1];
            Inter_L1_Shift[bi*4+0] = 3'd1;
        end else begin
            Inter_L1_ES[bi*4+0]    = Page_Inter_ES[pair_base+0];
            Inter_L1_Shift[bi*4+0] = 3'd0;
        end

        if (Page_Inter_ES[pair_base+3] < Page_Inter_ES[pair_base+2]) begin
            Inter_L1_ES[bi*4+1]    = Page_Inter_ES[pair_base+3];
            Inter_L1_Shift[bi*4+1] = 3'd3;
        end else begin
            Inter_L1_ES[bi*4+1]    = Page_Inter_ES[pair_base+2];
            Inter_L1_Shift[bi*4+1] = 3'd2;
        end

        if (Page_Inter_ES[pair_base+5] < Page_Inter_ES[pair_base+4]) begin
            Inter_L1_ES[bi*4+2]    = Page_Inter_ES[pair_base+5];
            Inter_L1_Shift[bi*4+2] = 3'd5;
        end else begin
            Inter_L1_ES[bi*4+2]    = Page_Inter_ES[pair_base+4];
            Inter_L1_Shift[bi*4+2] = 3'd4;
        end

        if (Page_Inter_ES[pair_base+7] < Page_Inter_ES[pair_base+6]) begin
            Inter_L1_ES[bi*4+3]    = Page_Inter_ES[pair_base+7];
            Inter_L1_Shift[bi*4+3] = 3'd7;
        end else begin
            Inter_L1_ES[bi*4+3]    = Page_Inter_ES[pair_base+6];
            Inter_L1_Shift[bi*4+3] = 3'd6;
        end

        // Level 2: 4 -> 2
        if (Inter_L1_ES[bi*4+1] < Inter_L1_ES[bi*4+0]) begin
            Inter_L2_ES[bi*2+0]    = Inter_L1_ES[bi*4+1];
            Inter_L2_Shift[bi*2+0] = Inter_L1_Shift[bi*4+1];
        end else begin
            Inter_L2_ES[bi*2+0]    = Inter_L1_ES[bi*4+0];
            Inter_L2_Shift[bi*2+0] = Inter_L1_Shift[bi*4+0];
        end

        if (Inter_L1_ES[bi*4+3] < Inter_L1_ES[bi*4+2]) begin
            Inter_L2_ES[bi*2+1]    = Inter_L1_ES[bi*4+3];
            Inter_L2_Shift[bi*2+1] = Inter_L1_Shift[bi*4+3];
        end else begin
            Inter_L2_ES[bi*2+1]    = Inter_L1_ES[bi*4+2];
            Inter_L2_Shift[bi*2+1] = Inter_L1_Shift[bi*4+2];
        end

        // Level 3: 2 -> 1
        if (Inter_L2_ES[bi*2+1] < Inter_L2_ES[bi*2+0]) begin
            Bank_Best_ES_Comb[bi]    = Inter_L2_ES[bi*2+1];
            Bank_Best_Shift_Comb[bi] = Inter_L2_Shift[bi*2+1];
        end else begin
            Bank_Best_ES_Comb[bi]    = Inter_L2_ES[bi*2+0];
            Bank_Best_Shift_Comb[bi] = Inter_L2_Shift[bi*2+0];
        end
    end
end

reg                     S5_Bank_Min_Valid;
reg [PAGE_ES_WIDTH-1:0] Bank_Best_ES_Reg [0:BANKS-1];
reg [CW_LENGTH-1:0]     Bank_Best_Shift_Buffer [0:BANKS-1];
reg [PAGE_ES_WIDTH-1:0] Final_Base_ES_Reg;
reg [PAGE_ES_WIDTH-1:0] Final_Intra_ES_Reg;

// -------------------------------------------------------------------------
// S6 : balanced 8-Bank adder tree only.
//      Input is the registered S5 Bank minimum values.
// -------------------------------------------------------------------------
wire [PAGE_ES_WIDTH-1:0] Bank_Sum_01_Reg = Bank_Best_ES_Reg[0] + Bank_Best_ES_Reg[1];
wire [PAGE_ES_WIDTH-1:0] Bank_Sum_23_Reg = Bank_Best_ES_Reg[2] + Bank_Best_ES_Reg[3];
wire [PAGE_ES_WIDTH-1:0] Bank_Sum_45_Reg = Bank_Best_ES_Reg[4] + Bank_Best_ES_Reg[5];
wire [PAGE_ES_WIDTH-1:0] Bank_Sum_67_Reg = Bank_Best_ES_Reg[6] + Bank_Best_ES_Reg[7];
wire [PAGE_ES_WIDTH-1:0] Bank_Sum_0123_Reg = Bank_Sum_01_Reg + Bank_Sum_23_Reg;
wire [PAGE_ES_WIDTH-1:0] Bank_Sum_4567_Reg = Bank_Sum_45_Reg + Bank_Sum_67_Reg;
wire [PAGE_ES_WIDTH-1:0] Page_Inter_Total_Wire = Bank_Sum_0123_Reg + Bank_Sum_4567_Reg;

reg                     S6_Inter_Sum_Valid;
reg [PAGE_ES_WIDTH-1:0] Final_Inter_ES_Reg;

// -------------------------------------------------------------------------
// S7 : final page-mode compare only.
//
// Tie policy is unchanged:
//   * INTER wins a tie with INTRA.
//   * BASE wins unless a remap candidate is strictly better than BASE.
// -------------------------------------------------------------------------
wire Final_Inter_Better_Or_Equal =
    (Final_Inter_ES_Reg <= Final_Intra_ES_Reg);
wire [PAGE_ES_WIDTH-1:0] Final_Best_Remap_ES =
    Final_Inter_Better_Or_Equal ? Final_Inter_ES_Reg : Final_Intra_ES_Reg;
wire [1:0] Final_Best_Remap_Mode =
    Final_Inter_Better_Or_Equal ? REMAP_INTER_BANK : REMAP_INTRA_BANK;
wire [1:0] Final_Page_Selected_Mode =
    (Final_Best_Remap_ES < Final_Base_ES_Reg) ?
        Final_Best_Remap_Mode : REMAP_NONE;

// -------------------------------------------------------------------------
// S8 : registered final result + sequential FICAM write-back.
// -------------------------------------------------------------------------
reg                       Writeback_Active;
reg [COUNT_WIDTH-1:0]     Write_Count;
reg [COUNT_WIDTH-1:0]     Write_Pointer;
reg [1:0]                 Page_Selected_Mode_Reg;

always @(*) begin
    Config_Write_Enable = Writeback_Active;
    Config_Index        = Page_Index_Buffer[Write_Pointer];
    Config_Remap_Mode   = Page_Selected_Mode_Reg;

    case (Page_Selected_Mode_Reg)
        REMAP_INTER_BANK:
            Config_Remap_Value =
                Bank_Best_Shift_Buffer[Page_Bank_Buffer[Write_Pointer]];

        REMAP_INTRA_BANK:
            Config_Remap_Value = Page_CW_Buffer[Write_Pointer];

        default:
            Config_Remap_Value = {CW_LENGTH{1'b0}};
    endcase
end

assign Busy = Fetch_Active | S1_Valid | S2_Valid | S3A_Valid | S3_Valid |
              Page_Closing | Finalize_Pending |
              S5_Bank_Min_Valid | S6_Inter_Sum_Valid |
              Writeback_Active;

integer r;
integer k;
integer bacc;
always @(posedge clk) begin
    if (!rst_n) begin
        Fetch_Active       <= 1'b0;
        Fetch_Index        <= {INDEX_WIDTH{1'b0}};
        Eval_Count_Latched <= {COUNT_WIDTH{1'b0}};
        Page_Closing       <= 1'b0;

        S1_Valid       <= 1'b0;
        S1_Index       <= {INDEX_WIDTH{1'b0}};
        S1_Bank        <= {BANK_WIDTH{1'b0}};
        S1_FI_Valid    <= {FI_PER_ENTRY{1'b0}};
        S1_FI_FP       <= {(FI_PER_ENTRY*FP_LENGTH){1'b0}};
        S1_Is_Exp      <= 1'b0;
        S1_Page_Last   <= 1'b0;

        S2_Valid       <= 1'b0;
        S2_Index       <= {INDEX_WIDTH{1'b0}};
        S2_Bank        <= {BANK_WIDTH{1'b0}};
        S2_Base_ES     <= {ES_WIDTH{1'b0}};
        S2_Intra_ES    <= {(CANDIDATES*ES_WIDTH){1'b0}};
        S2_Inter_ES    <= {(CANDIDATES*ES_WIDTH){1'b0}};
        S2_Page_Last   <= 1'b0;

        S3A_Valid       <= 1'b0;
        S3A_Index       <= {INDEX_WIDTH{1'b0}};
        S3A_Bank        <= {BANK_WIDTH{1'b0}};
        S3A_Base_ES     <= {ES_WIDTH{1'b0}};
        S3A_I01_ES      <= {ES_WIDTH{1'b0}};
        S3A_I23_ES      <= {ES_WIDTH{1'b0}};
        S3A_I45_ES      <= {ES_WIDTH{1'b0}};
        S3A_I67_ES      <= {ES_WIDTH{1'b0}};
        S3A_I01_IDX     <= {CW_LENGTH{1'b0}};
        S3A_I23_IDX     <= {CW_LENGTH{1'b0}};
        S3A_I45_IDX     <= {CW_LENGTH{1'b0}};
        S3A_I67_IDX     <= {CW_LENGTH{1'b0}};
        S3A_Inter_ES    <= {(CANDIDATES*ES_WIDTH){1'b0}};
        S3A_Page_Last   <= 1'b0;

        S3_Valid         <= 1'b0;
        S3_Index         <= {INDEX_WIDTH{1'b0}};
        S3_Bank          <= {BANK_WIDTH{1'b0}};
        S3_Bank_OneHot   <= {BANKS{1'b0}};
        S3_Base_ES       <= {ES_WIDTH{1'b0}};
        S3_Best_Intra_ES <= {ES_WIDTH{1'b0}};
        S3_Best_CW       <= {CW_LENGTH{1'b0}};
        S3_Inter_ES      <= {(CANDIDATES*ES_WIDTH){1'b0}};
        S3_Page_Last     <= 1'b0;

        Page_Base_ES     <= {PAGE_ES_WIDTH{1'b0}};
        Page_Intra_ES    <= {PAGE_ES_WIDTH{1'b0}};
        Page_Entry_Count <= {COUNT_WIDTH{1'b0}};
        Finalize_Pending <= 1'b0;

        S5_Bank_Min_Valid <= 1'b0;
        Final_Base_ES_Reg <= {PAGE_ES_WIDTH{1'b0}};
        Final_Intra_ES_Reg <= {PAGE_ES_WIDTH{1'b0}};

        S6_Inter_Sum_Valid <= 1'b0;
        Final_Inter_ES_Reg <= {PAGE_ES_WIDTH{1'b0}};

        Writeback_Active       <= 1'b0;
        Write_Count            <= {COUNT_WIDTH{1'b0}};
        Write_Pointer          <= {COUNT_WIDTH{1'b0}};
        Page_Selected_Mode_Reg <= REMAP_NONE;

        Remap_Mode <= REMAP_NONE;
        Eval_Done  <= 1'b0;

        for (r = 0; r < INTER_ACCUMS; r = r + 1)
            Page_Inter_ES[r] <= {PAGE_ES_WIDTH{1'b0}};

        for (r = 0; r < FICAM_ENTRIES; r = r + 1) begin
            Page_Index_Buffer[r] <= {INDEX_WIDTH{1'b0}};
            Page_Bank_Buffer[r]  <= {BANK_WIDTH{1'b0}};
            Page_CW_Buffer[r]    <= {CW_LENGTH{1'b0}};
        end

        for (r = 0; r < BANKS; r = r + 1) begin
            Bank_Best_ES_Reg[r]         <= {PAGE_ES_WIDTH{1'b0}};
            Bank_Best_Shift_Buffer[r]   <= {CW_LENGTH{1'b0}};
        end
    end
    else begin
        Eval_Done <= 1'b0;

        if (Eval_Start && !Busy) begin
            Remap_Mode          <= REMAP_NONE;
            Eval_Count_Latched  <= Eval_Entry_Count;
            Fetch_Index         <= {INDEX_WIDTH{1'b0}};
            Page_Closing        <= 1'b0;
            Page_Base_ES        <= {PAGE_ES_WIDTH{1'b0}};
            Page_Intra_ES       <= {PAGE_ES_WIDTH{1'b0}};
            Page_Entry_Count    <= {COUNT_WIDTH{1'b0}};
            Finalize_Pending    <= 1'b0;
            S5_Bank_Min_Valid   <= 1'b0;
            S6_Inter_Sum_Valid  <= 1'b0;
            Writeback_Active    <= 1'b0;
            Write_Count         <= {COUNT_WIDTH{1'b0}};
            Write_Pointer       <= {COUNT_WIDTH{1'b0}};
            S1_Valid            <= 1'b0;
            S2_Valid            <= 1'b0;
            S3A_Valid           <= 1'b0;
            S3_Valid            <= 1'b0;
            S3_Bank_OneHot      <= {BANKS{1'b0}};
            Final_Base_ES_Reg   <= {PAGE_ES_WIDTH{1'b0}};
            Final_Intra_ES_Reg  <= {PAGE_ES_WIDTH{1'b0}};
            Final_Inter_ES_Reg  <= {PAGE_ES_WIDTH{1'b0}};

            for (k = 0; k < INTER_ACCUMS; k = k + 1)
                Page_Inter_ES[k] <= {PAGE_ES_WIDTH{1'b0}};

            for (k = 0; k < BANKS; k = k + 1) begin
                Bank_Best_ES_Reg[k]       <= {PAGE_ES_WIDTH{1'b0}};
                Bank_Best_Shift_Buffer[k] <= {CW_LENGTH{1'b0}};
            end

            if (Eval_Entry_Count == {COUNT_WIDTH{1'b0}}) begin
                Fetch_Active <= 1'b0;
                Eval_Done    <= 1'b1;
            end
            else begin
                Fetch_Active <= 1'b1;
            end
        end
        else begin
            // S5/S6 valid flags are one-cycle pipeline pulses.
            S5_Bank_Min_Valid  <= 1'b0;
            S6_Inter_Sum_Valid <= 1'b0;

            // ---------------- S1 ----------------
            S1_Valid <= Fetch_Active && Eval_Read_Valid;
            if (Fetch_Active && Eval_Read_Valid) begin
                S1_Index     <= Fetch_Index;
                S1_Bank      <= Eval_Read_Bank;
                S1_FI_Valid  <= Eval_Read_FI_Valid;
                S1_FI_FP     <= Eval_Read_FI_FP;
                S1_Is_Exp    <= Eval_Read_Is_Exp;
                S1_Page_Last <= (Fetch_Position_Next >= Eval_Count_Latched);

                if (Fetch_Position_Next >= Eval_Count_Latched) begin
                    Fetch_Active <= 1'b0;
                    Page_Closing <= 1'b1;
                end
                else begin
                    Fetch_Index <= Fetch_Index + 1'b1;
                end
            end

            // ---------------- S2 ----------------
            S2_Valid <= S1_Valid;
            if (S1_Valid) begin
                S2_Index     <= S1_Index;
                S2_Bank      <= S1_Bank;
                S2_Base_ES   <= Base_ES_Wire;
                S2_Intra_ES  <= Intra_ES_Wire;
                S2_Inter_ES  <= Inter_ES_Wire;
                S2_Page_Last <= S1_Page_Last;
            end

            // ---------------- S3A ----------------
            // Only the first pairwise comparator level is placed here.
            S3A_Valid <= S2_Valid;
            if (S2_Valid) begin
                S3A_Index     <= S2_Index;
                S3A_Bank      <= S2_Bank;
                S3A_Base_ES   <= S2_Base_ES;
                S3A_I01_ES    <= I01;
                S3A_I23_ES    <= I23;
                S3A_I45_ES    <= I45;
                S3A_I67_ES    <= I67;
                S3A_I01_IDX   <= I01_IDX;
                S3A_I23_IDX   <= I23_IDX;
                S3A_I45_IDX   <= I45_IDX;
                S3A_I67_IDX   <= I67_IDX;
                S3A_Inter_ES  <= S2_Inter_ES;
                S3A_Page_Last <= S2_Page_Last;
            end

            // ---------------- S3B ----------------
            // The remaining two comparator levels generate the final RG CW.
            S3_Valid <= S3A_Valid;
            if (S3A_Valid) begin
                S3_Index         <= S3A_Index;
                S3_Bank          <= S3A_Bank;
                S3_Bank_OneHot   <= S3_Bank_OneHot_Wire;
                S3_Base_ES       <= S3A_Base_ES;
                S3_Best_Intra_ES <= Best_Intra_ES_Wire;
                S3_Best_CW       <= Best_Intra_CW_Wire;
                S3_Inter_ES      <= S3A_Inter_ES;
                S3_Page_Last     <= S3A_Page_Last;
            end

            // ---------------- S4 ----------------
            // Every entry, including the last one, is first accumulated into
            // page/bank registers. Finalize_Pending delays S5 by one clock so
            // S5 sees the complete page totals.
            if (S3_Valid && !Writeback_Active && !Finalize_Pending) begin
                Page_Index_Buffer[Page_Entry_Count] <= S3_Index;
                Page_Bank_Buffer[Page_Entry_Count]  <= S3_Bank;
                Page_CW_Buffer[Page_Entry_Count]    <= S3_Best_CW;

                Page_Base_ES     <= Page_Base_ES + S4_Base_Ext;
                Page_Intra_ES    <= Page_Intra_ES + S4_Intra_Ext;
                Page_Entry_Count <= Page_Entry_Count + 1'b1;

                // Fixed-index Bank accumulation.  Each synthesized Bank has
                // its own registered one-hot write enable, removing the
                // S3_Bank binary decode / dynamic index from this path.
                for (bacc = 0; bacc < BANKS; bacc = bacc + 1) begin
                    if (S3_Bank_OneHot[bacc]) begin
                        for (k = 0; k < CANDIDATES; k = k + 1) begin
                            Page_Inter_ES[bacc*CANDIDATES + k] <=
                                Page_Inter_ES[bacc*CANDIDATES + k] +
                                {{(PAGE_ES_WIDTH-ES_WIDTH){1'b0}},
                                 S3_Inter_ES[k*ES_WIDTH +: ES_WIDTH]};
                        end
                    end
                end

                if (S3_Page_Last)
                    Finalize_Pending <= 1'b1;
            end

            // ---------------- S5 ----------------
            // Per-Bank minimum only. The complete Base/Intra page totals and
            // write-back count are latched here as well.
            if (Finalize_Pending && !Writeback_Active) begin
                Final_Base_ES_Reg  <= Page_Base_ES;
                Final_Intra_ES_Reg <= Page_Intra_ES;
                Write_Count        <= Page_Entry_Count;
                Write_Pointer      <= {COUNT_WIDTH{1'b0}};

                for (k = 0; k < BANKS; k = k + 1) begin
                    Bank_Best_ES_Reg[k]       <= Bank_Best_ES_Comb[k];
                    Bank_Best_Shift_Buffer[k] <= Bank_Best_Shift_Comb[k];
                end

                S5_Bank_Min_Valid <= 1'b1;
                Finalize_Pending  <= 1'b0;

                // Page metadata buffers are retained through S8 write-back.
                // Accumulators can now be cleared because their final values
                // have been captured by the S5 registers.
                Page_Base_ES     <= {PAGE_ES_WIDTH{1'b0}};
                Page_Intra_ES    <= {PAGE_ES_WIDTH{1'b0}};
                Page_Entry_Count <= {COUNT_WIDTH{1'b0}};
                for (k = 0; k < INTER_ACCUMS; k = k + 1)
                    Page_Inter_ES[k] <= {PAGE_ES_WIDTH{1'b0}};
            end

            // ---------------- S6 ----------------
            // Balanced Bank sum only. This register cuts the path between the
            // per-Bank minimum tree and final page-mode comparison.
            if (S5_Bank_Min_Valid) begin
                Final_Inter_ES_Reg  <= Page_Inter_Total_Wire;
                S6_Inter_Sum_Valid  <= 1'b1;
            end

            // ---------------- S7 ----------------
            // Final Base / Inter / Intra comparison only.
            if (S6_Inter_Sum_Valid && !Writeback_Active) begin
                Page_Selected_Mode_Reg <= Final_Page_Selected_Mode;
                Remap_Mode             <= Final_Page_Selected_Mode;
                Write_Pointer          <= {COUNT_WIDTH{1'b0}};
                Writeback_Active       <= 1'b1;
            end

            // ---------------- S8 ----------------
            // FICAM samples Config_* on the active edge, then the pointer moves.
            if (Writeback_Active) begin
                if ((Write_Pointer + 1'b1) >= Write_Count) begin
                    Writeback_Active <= 1'b0;
                    Page_Closing     <= 1'b0;
                    Write_Pointer    <= {COUNT_WIDTH{1'b0}};
                    Eval_Done        <= 1'b1;
                end
                else begin
                    Write_Pointer <= Write_Pointer + 1'b1;
                end
            end
        end
    end
end

endmodule
