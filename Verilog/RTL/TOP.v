// External sequence:
//   1) rst_n=0 -> 1
//   2) write 0..16 packed FICAM_Entry records
//   3) pulse Eval_Start for one cycle
//   4) CU evaluates all currently loaded entries as one page
//   5) remap results are written back to FICAM; normal SRAM writes can follow

module TOP #(
    parameter integer PPN_LENGTH         = 16,
    parameter integer BANK_LENGTH        = 3,
    parameter integer RG_LENGTH          = 3,
    parameter integer FW_LENGTH          = 3,
    parameter integer FP_LENGTH          = 3,
    parameter integer CW_LENGTH          = 3,
    parameter integer FICAM_ENTRIES      = 64,
    parameter integer FI_PER_ENTRY       = 2,

    parameter integer SRAM_ADDR_LENGTH   = 14,
    parameter integer WORD_ADDR_LENGTH   = 3,
    parameter integer ES_WIDTH           = 16,
    parameter integer DATA_WIDTH         = 8,
    parameter integer WW_LENGTH          = DATA_WIDTH,
    parameter integer TWW_LENGTH         = DATA_WIDTH,

    parameter integer FI_SLOT_WIDTH      = 1 + FW_LENGTH + FP_LENGTH,
    parameter integer FICAM_LENGTH       = 1 + PPN_LENGTH + BANK_LENGTH + RG_LENGTH
                                           + FI_PER_ENTRY*FI_SLOT_WIDTH,
    parameter integer FICAM_INDEX_WIDTH  = (FICAM_ENTRIES <= 2) ? 1 : $clog2(FICAM_ENTRIES),
    parameter integer FICAM_COUNT_WIDTH  = (FICAM_ENTRIES <= 1) ? 1 : $clog2(FICAM_ENTRIES + 1),
    parameter integer DATA_PACKET_LENGTH = 1 + PPN_LENGTH + BANK_LENGTH + RG_LENGTH
                                           + SRAM_ADDR_LENGTH + DATA_WIDTH
)(
    input  wire                              clk,
    input  wire                              rst_n,

    // LOAD phase input.
    input  wire                              Group_Is_Exp,
    input  wire [FICAM_LENGTH-1:0]           FICAM_Entry,

    // One-cycle command that freezes FI loading and begins page evaluation.
    input  wire                              Eval_Start,

    // Normal write packet: {Valid, PPN, Bank, RG, SRAM_Logical_Address, Data}.
    input  wire [DATA_PACKET_LENGTH-1:0]      Data_Write_Packet,
    output wire                              Data_Ready,

    output wire [1:0]                        Remap_Mode
);

localparam [1:0] REMAP_NONE       = 2'b00;
localparam [1:0] REMAP_INTER_BANK = 2'b01;
localparam [1:0] REMAP_INTRA_BANK = 2'b10;

// -------------------------------------------------------------------------
// LOAD/EVAL phase state.
// -------------------------------------------------------------------------
reg evaluation_phase;
wire eval_start_accept;
wire cu_busy;
wire cu_eval_done;

assign eval_start_accept = Eval_Start && !evaluation_phase && !cu_busy;

always @(posedge clk) begin
    if (!rst_n)
        evaluation_phase <= 1'b0;
    else if (eval_start_accept)
        evaluation_phase <= 1'b1;
end

// -------------------------------------------------------------------------
// FICAM <-> CU direct indexed evaluation interface.
// -------------------------------------------------------------------------
wire                                  ficam_full;
wire [FICAM_COUNT_WIDTH-1:0]          ficam_entry_count;
wire                                  eval_read_enable;
wire [FICAM_INDEX_WIDTH-1:0]          eval_read_index;
wire                                  eval_read_valid;
wire [BANK_LENGTH-1:0]                eval_read_bank;
wire [FI_PER_ENTRY-1:0]               eval_read_fi_valid;
wire [FI_PER_ENTRY*FP_LENGTH-1:0]     eval_read_fi_fp;
wire                                  eval_read_is_exp;

wire                                  config_we;
wire [FICAM_INDEX_WIDTH-1:0]          config_index;
wire [1:0]                            config_mode;
wire [CW_LENGTH-1:0]                  config_value;

// -------------------------------------------------------------------------
// Packed normal-write input decode.
// -------------------------------------------------------------------------
localparam integer D_DATA_LSB = 0;
localparam integer D_ADDR_LSB = D_DATA_LSB + DATA_WIDTH;
localparam integer D_RG_LSB   = D_ADDR_LSB + SRAM_ADDR_LENGTH;
localparam integer D_BANK_LSB = D_RG_LSB + RG_LENGTH;
localparam integer D_PPN_LSB  = D_BANK_LSB + BANK_LENGTH;
localparam integer D_VALID_BIT= D_PPN_LSB + PPN_LENGTH;

wire                              data_valid;
wire [DATA_WIDTH-1:0]             data_in;
wire [SRAM_ADDR_LENGTH-1:0]       data_addr;
wire [RG_LENGTH-1:0]              data_rg;
wire [BANK_LENGTH-1:0]            data_bank;
wire [PPN_LENGTH-1:0]             data_ppn;

assign data_in    = Data_Write_Packet[D_DATA_LSB +: DATA_WIDTH];
assign data_addr  = Data_Write_Packet[D_ADDR_LSB +: SRAM_ADDR_LENGTH];
assign data_rg    = Data_Write_Packet[D_RG_LSB   +: RG_LENGTH];
assign data_bank  = Data_Write_Packet[D_BANK_LSB +: BANK_LENGTH];
assign data_ppn   = Data_Write_Packet[D_PPN_LSB  +: PPN_LENGTH];
assign data_valid = Data_Write_Packet[D_VALID_BIT];

wire                    search_match;
wire [1:0]              stored_mode;
wire [CW_LENGTH-1:0]    stored_value;

FICAM #(
    .PPN_LENGTH     (PPN_LENGTH),
    .BANK_LENGTH    (BANK_LENGTH),
    .RG_LENGTH      (RG_LENGTH),
    .FW_LENGTH      (FW_LENGTH),
    .FP_LENGTH      (FP_LENGTH),
    .CW_LENGTH      (CW_LENGTH),
    .FICAM_ENTRIES  (FICAM_ENTRIES),
    .FI_PER_ENTRY   (FI_PER_ENTRY),
    .FICAM_LENGTH   (FICAM_LENGTH),
    .INDEX_WIDTH    (FICAM_INDEX_WIDTH),
    .COUNT_WIDTH    (FICAM_COUNT_WIDTH)
) u_ficam (
    .clk                 (clk),
    .rst_n               (rst_n),
    .Group_Is_Exp        (Group_Is_Exp),
    .FICAM_Entry         (FICAM_Entry),
    .Load_Lock           (evaluation_phase),
    .FICAM_Full          (ficam_full),
    .Entry_Count_Out     (ficam_entry_count),

    .Eval_Read_Enable    (eval_read_enable),
    .Eval_Read_Index     (eval_read_index),
    .Eval_Read_Valid     (eval_read_valid),
    .Eval_Read_Bank      (eval_read_bank),
    .Eval_Read_FI_Valid  (eval_read_fi_valid),
    .Eval_Read_FI_FP     (eval_read_fi_fp),
    .Eval_Read_Is_Exp    (eval_read_is_exp),

    .Config_Write_Enable (config_we),
    .Config_Index        (config_index),
    .Config_Remap_Mode   (config_mode),
    .Config_Remap_Value  (config_value),

    .Search_Enable       (data_valid && evaluation_phase && !cu_busy),
    .Search_PPN          (data_ppn),
    .Search_Bank         (data_bank),
    .Search_RG           (data_rg),
    .Search_Match        (search_match),
    .Remap_Mode_Out      (stored_mode),
    .Remap_Value_Out     (stored_value)
);

Control_Unit #(
    .FP_LENGTH      (FP_LENGTH),
    .CW_LENGTH      (CW_LENGTH),
    .ES_WIDTH       (ES_WIDTH),
    .FI_PER_ENTRY   (FI_PER_ENTRY),
    .FICAM_ENTRIES  (FICAM_ENTRIES),
    .BANKS          (1 << BANK_LENGTH),
    .INDEX_WIDTH    (FICAM_INDEX_WIDTH),
    .COUNT_WIDTH    (FICAM_COUNT_WIDTH),
    .BANK_WIDTH     (BANK_LENGTH)
) u_control_unit (
    .clk                 (clk),
    .rst_n               (rst_n),
    .Eval_Start          (eval_start_accept),
    .Eval_Entry_Count    (ficam_entry_count),

    .Eval_Read_Enable    (eval_read_enable),
    .Eval_Read_Index     (eval_read_index),
    .Eval_Read_Valid     (eval_read_valid),
    .Eval_Read_Bank      (eval_read_bank),
    .Eval_Read_FI_Valid  (eval_read_fi_valid),
    .Eval_Read_FI_FP     (eval_read_fi_fp),
    .Eval_Read_Is_Exp    (eval_read_is_exp),

    .Config_Write_Enable (config_we),
    .Config_Index        (config_index),
    .Config_Remap_Mode   (config_mode),
    .Config_Remap_Value  (config_value),
    .Remap_Mode          (Remap_Mode),
    .Busy                (cu_busy),
    .Eval_Done           (cu_eval_done)
);

// -------------------------------------------------------------------------
// Write-only normal data path to SRAM (kept from the previous version).
// Normal data is stalled while ES evaluation/write-back is active.
// -------------------------------------------------------------------------
wire normal_path_enable;
assign normal_path_enable = evaluation_phase && !cu_busy;

wire [DATA_WIDTH-1:0] shifted_inter_data;
wire [DATA_WIDTH-1:0] unused_unshifted_data;
Data_Shifter #(
    .SM_LENGTH  (CW_LENGTH),
    .DATA_LENGTH(DATA_WIDTH)
) u_inter_shifter (
    .Shift_Mode       (stored_value),
    .Switched_Data_In (data_in),
    .Shifted_Data_In  ({DATA_WIDTH{1'b0}}),
    .Switched_Data_Out(unused_unshifted_data),
    .Shifted_Data_Out (shifted_inter_data)
);

localparam [1:0] INTRA_IDLE  = 2'd0;
localparam [1:0] INTRA_FIRST = 2'd1;
localparam [1:0] INTRA_LOAD  = 2'd2;
localparam [1:0] INTRA_DRAIN = 2'd3;

reg [1:0] intra_state;
reg [DATA_WIDTH-1:0] first_ww;
reg [3:0] intra_input_count;
reg [WORD_ADDR_LENGTH-1:0] intra_output_count;
reg [CW_LENGTH-1:0] intra_cw;
reg [SRAM_ADDR_LENGTH-WORD_ADDR_LENGTH-1:0] intra_addr_base;

wire trans_start;
assign trans_start = (intra_state == INTRA_IDLE) && normal_path_enable &&
                     data_valid && search_match &&
                     (stored_mode == REMAP_INTRA_BANK);
wire trans_input_ready;
wire trans_output_valid;
wire trans_busy;
wire trans_done;
wire [TWW_LENGTH-1:0] transposed_data;
wire [WW_LENGTH-1:0] unused_ww_out;

wire feed_cached_first;
wire feed_external_ww;
wire trans_data_valid;
wire [WW_LENGTH-1:0] trans_ww_in;
assign feed_cached_first = (intra_state == INTRA_FIRST) && trans_input_ready;
assign feed_external_ww  = (intra_state == INTRA_LOAD) && trans_input_ready &&
                           normal_path_enable && data_valid;
assign trans_data_valid  = feed_cached_first || feed_external_ww;
assign trans_ww_in       = feed_cached_first ? first_ww : data_in;

Transposer #(
    .WORD_COUNT  (8),
    .WW_LENGTH   (WW_LENGTH),
    .TWW_LENGTH  (TWW_LENGTH),
    .COUNT_LENGTH(WORD_ADDR_LENGTH)
) u_transposer (
    .clk           (clk),
    .rst_n         (rst_n),
    .Start         (trans_start),
    .Direction     (1'b1),
    .Data_In_Valid (trans_data_valid),
    .Input_Ready   (trans_input_ready),
    .WW_Data_In    (trans_ww_in),
    .TWW_Data_In   ({TWW_LENGTH{1'b0}}),
    .WW_Data_Out   (unused_ww_out),
    .TWW_Data_Out  (transposed_data),
    .Data_Out_Valid(trans_output_valid),
    .Busy          (trans_busy),
    .Done          (trans_done)
);

wire [WORD_ADDR_LENGTH-1:0] intra_physical_word;
Address_Remapper #(
    .WORD_ADDR_LENGTH(WORD_ADDR_LENGTH),
    .CW_LENGTH       (CW_LENGTH)
) u_intra_address_remapper (
    .Remap_Enable      (1'b1),
    .Control_Word      (intra_cw),
    .Logical_Word_Addr (intra_output_count),
    .Physical_Word_Addr(intra_physical_word)
);

assign Data_Ready = normal_path_enable &&
                    (((intra_state == INTRA_IDLE)) ||
                     ((intra_state == INTRA_LOAD) && trans_input_ready));

always @(posedge clk) begin
    if (!rst_n) begin
        intra_state        <= INTRA_IDLE;
        first_ww           <= {DATA_WIDTH{1'b0}};
        intra_input_count  <= 4'd0;
        intra_output_count <= {WORD_ADDR_LENGTH{1'b0}};
        intra_cw           <= {CW_LENGTH{1'b0}};
        intra_addr_base    <= {(SRAM_ADDR_LENGTH-WORD_ADDR_LENGTH){1'b0}};
    end
    else begin
        case (intra_state)
            INTRA_IDLE: begin
                intra_input_count  <= 4'd0;
                intra_output_count <= {WORD_ADDR_LENGTH{1'b0}};
                if (trans_start) begin
                    first_ww        <= data_in;
                    intra_cw        <= stored_value;
                    intra_addr_base <= data_addr[SRAM_ADDR_LENGTH-1:WORD_ADDR_LENGTH];
                    intra_state     <= INTRA_FIRST;
                end
            end

            INTRA_FIRST: begin
                if (feed_cached_first) begin
                    intra_input_count <= 4'd1;
                    intra_state       <= INTRA_LOAD;
                end
            end

            INTRA_LOAD: begin
                if (feed_external_ww) begin
                    if (intra_input_count == 4'd7) begin
                        intra_input_count <= 4'd8;
                        intra_state       <= INTRA_DRAIN;
                    end
                    else begin
                        intra_input_count <= intra_input_count + 1'b1;
                    end
                end
            end

            INTRA_DRAIN: begin
                if (trans_output_valid) begin
                    if (intra_output_count == 3'd7)
                        intra_output_count <= {WORD_ADDR_LENGTH{1'b0}};
                    else
                        intra_output_count <= intra_output_count + 1'b1;
                end
                if (trans_done)
                    intra_state <= INTRA_IDLE;
            end

            default: intra_state <= INTRA_IDLE;
        endcase
    end
end

wire direct_write;
wire intra_write;
wire sram_write_enable;
wire [SRAM_ADDR_LENGTH-1:0] direct_addr;
wire [DATA_WIDTH-1:0] direct_data;
wire [SRAM_ADDR_LENGTH-1:0] intra_addr;
wire [SRAM_ADDR_LENGTH-1:0] sram_addr;
wire [DATA_WIDTH-1:0] sram_data;

assign direct_write = normal_path_enable && data_valid &&
                      (intra_state == INTRA_IDLE) &&
                      !(search_match && (stored_mode == REMAP_INTRA_BANK));
assign intra_write  = (intra_state == INTRA_DRAIN) && trans_output_valid;
assign sram_write_enable = direct_write || intra_write;
assign direct_addr = data_addr;
assign direct_data = (search_match && (stored_mode == REMAP_INTER_BANK))
                   ? shifted_inter_data : data_in;
assign intra_addr = {intra_addr_base, intra_physical_word};
assign sram_addr = intra_write ? intra_addr : direct_addr;
assign sram_data = intra_write ? transposed_data : direct_data;

wire [DATA_WIDTH-1:0] sram_q_unused;
SRAM_SP_ADV_rtl_top u_sram (
    .Q   (sram_q_unused),
    .CLK (clk),
    .CEN (~sram_write_enable),
    .WEN (~sram_write_enable),
    .A   (sram_addr),
    .D   (sram_data),
    .EMA (3'b000)
);

wire _unused_top;
assign _unused_top = &{1'b0, ficam_full, cu_eval_done, trans_busy};

endmodule
