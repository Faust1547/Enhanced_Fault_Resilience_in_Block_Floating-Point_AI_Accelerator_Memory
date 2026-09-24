`timescale 1ns / 1ps

module TOP_tb;
localparam PPN_LENGTH    = 16;
localparam BANK_LENGTH   = 3;
localparam RG_LENGTH     = 3;
localparam FW_LENGTH     = 3;
localparam FP_LENGTH     = 3;
localparam CW_LENGTH     = 3;
localparam FI_PER_ENTRY  = 2;
localparam FI_SLOT_WIDTH = 1 + FW_LENGTH + FP_LENGTH;
localparam FICAM_LENGTH  = 1 + PPN_LENGTH + BANK_LENGTH + RG_LENGTH
                           + FI_PER_ENTRY*FI_SLOT_WIDTH;
localparam SRAM_ADDR_LENGTH = 14;
localparam DATA_WIDTH = 8;
localparam DATA_PACKET_LENGTH = 1 + PPN_LENGTH + BANK_LENGTH + RG_LENGTH
                                + SRAM_ADDR_LENGTH + DATA_WIDTH;

reg clk;
reg rst_n;
reg Group_Is_Exp;
reg [FICAM_LENGTH-1:0] FICAM_Entry;
reg Eval_Start;
reg [DATA_PACKET_LENGTH-1:0] Data_Write_Packet;
wire Data_Ready;
wire [1:0] Remap_Mode;

TOP #(
    .PPN_LENGTH(PPN_LENGTH),
    .BANK_LENGTH(BANK_LENGTH),
    .RG_LENGTH(RG_LENGTH),
    .FW_LENGTH(FW_LENGTH),
    .FP_LENGTH(FP_LENGTH),
    .CW_LENGTH(CW_LENGTH),
    .FICAM_ENTRIES(16)
) dut (
    .clk(clk),
    .rst_n(rst_n),
    .Group_Is_Exp(Group_Is_Exp),
    .FICAM_Entry(FICAM_Entry),
    .Eval_Start(Eval_Start),
    .Data_Write_Packet(Data_Write_Packet),
    .Data_Ready(Data_Ready),
    .Remap_Mode(Remap_Mode)
);

always #5 clk = ~clk;

task Write_FICAM_Record;
    input [PPN_LENGTH-1:0]  ppn;
    input [BANK_LENGTH-1:0] bank;
    input [RG_LENGTH-1:0]   rg;
    input                   v0;
    input [FW_LENGTH-1:0]   fw0;
    input [FP_LENGTH-1:0]   fp0;
    input                   v1;
    input [FW_LENGTH-1:0]   fw1;
    input [FP_LENGTH-1:0]   fp1;
    begin
        @(negedge clk);
        FICAM_Entry = {1'b1, ppn, bank, rg,
                       v1, fw1, fp1,
                       v0, fw0, fp0};
        @(negedge clk);
        FICAM_Entry = {FICAM_LENGTH{1'b0}};
    end
endtask

task Pulse_Eval_Start;
    begin
        @(negedge clk);
        Eval_Start = 1'b1;
        @(negedge clk);
        Eval_Start = 1'b0;
    end
endtask

integer k;
initial begin
    clk = 1'b0;
    rst_n = 1'b0;
    Group_Is_Exp = 1'b0;
    FICAM_Entry = {FICAM_LENGTH{1'b0}};
    Eval_Start = 1'b0;
    Data_Write_Packet = {DATA_PACKET_LENGTH{1'b0}};

    repeat (3) @(posedge clk);
    @(negedge clk);
    rst_n = 1'b1;

    // ---------------- LOAD PHASE ----------------
    // S&M, Bank 2, RG 3, FP7 + FP6:
    // Base=48. Best Intra=4 (CW4). Best Inter=4 (Shift2 first min).
    // INTER wins the Intra tie because both improve over BASE.
    Write_FICAM_Record(16'h0021, 3'd2, 3'd3,
                       1'b1, 3'd1, 3'd7,
                       1'b1, 3'd5, 3'd6);

    if (dut.u_ficam.Entry_Count !== 1) begin
        $display("[FAIL] LOAD phase Entry_Count expected 1, got %0d",
                 dut.u_ficam.Entry_Count);
        $finish;
    end

    // ---------------- EVALUATION PHASE ----------------
    Pulse_Eval_Start();

    // Wait for the CU's internal one-cycle completion pulse.
    k = 0;
    while (!dut.cu_eval_done && (k < 50)) begin
        @(posedge clk);
        k = k + 1;
    end

    if (!dut.cu_eval_done) begin
        $display("[FAIL] evaluation timeout");
        $finish;
    end

    if (Remap_Mode !== 2'b01) begin
        $display("[FAIL] expected INTER mode, got %b", Remap_Mode);
        $finish;
    end

    if (dut.u_ficam.Mode_Mem[0] !== 2'b01 ||
        dut.u_ficam.Remap_Value_Mem[0] !== 3'd2) begin
        $display("[FAIL] FICAM write-back mismatch: mode=%b value=%0d",
                 dut.u_ficam.Mode_Mem[0], dut.u_ficam.Remap_Value_Mem[0]);
        $finish;
    end

    // Verify phase lock: FI writes after Eval_Start must be ignored.
    Write_FICAM_Record(16'h0021, 3'd5, 3'd1,
                       1'b1, 3'd0, 3'd4,
                       1'b0, 3'd0, 3'd0);

    repeat (2) @(posedge clk);
    if (dut.u_ficam.Entry_Count !== 1) begin
        $display("[FAIL] FI was accepted after evaluation lock");
        $finish;
    end

    $display("[PASS] LOAD -> Eval_Start -> indexed FICAM read -> 9-stage CU -> write-back");
    $finish;
end

endmodule
