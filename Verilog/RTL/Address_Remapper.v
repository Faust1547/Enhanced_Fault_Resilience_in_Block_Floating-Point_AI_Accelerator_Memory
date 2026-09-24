module Address_Remapper #(
    parameter integer WORD_ADDR_LENGTH = 3,
    parameter integer CW_LENGTH        = 3
)(
    input  wire                         Remap_Enable,
    input  wire [CW_LENGTH-1:0]         Control_Word,
    input  wire [WORD_ADDR_LENGTH-1:0]  Logical_Word_Addr,
    output wire [WORD_ADDR_LENGTH-1:0]  Physical_Word_Addr
);

assign Physical_Word_Addr = Remap_Enable
    ? (Logical_Word_Addr ^ Control_Word[WORD_ADDR_LENGTH-1:0])
    : Logical_Word_Addr;

endmodule
