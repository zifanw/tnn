// Testbench for a 16-input bitonic sorter - You don't have to modify it

// Input lines make 1->0 transitions at different times (essentially values
// ranging from 0 to 39) coming in at the 16 input lines in an unsorted manner.
// At the output of the sorter, the lines should have 1->0 transitions in
// ascending order.

`timescale 1ns / 1ps

module bitonic_sort_tb;

    reg [0:15] raw_in;
    wire [0:15] sorted_out;

    bitonic_sort_16 DUT (.sorted_out(sorted_out), .raw_in(raw_in));

    initial
    begin

        $dumpfile("bitonic_sort_16.vcd");
        $dumpvars(0, bitonic_sort_tb);

        raw_in = ~16'b0;

        #5
        raw_in[1] = 0;

        #2
        raw_in[13] = 0;

        #3
        raw_in[5] = 0;

        #1
        raw_in[12] = 0;

        #4
        raw_in[4] = 0;

        #2
        raw_in[7] = 0;

        #3
        raw_in[3] = 0;

        #1
        raw_in[15] = 0;

        #5
        raw_in[10] = 0;
        
        #4
        raw_in[2] = 0;
        
        #5
        raw_in[8] = 0;
        
        #2
        raw_in[11] = 0;
        
        #3
        raw_in[14] = 0;
        
        #1
        raw_in[0] = 0;
        
        #2
        raw_in[6] = 0;
        
        #1
        raw_in[9] = 0;
        
        #10
        raw_in = ~16'b0;

        #200
        $finish;

    end

endmodule
