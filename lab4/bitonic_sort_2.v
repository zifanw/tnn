// A 2-input bitonic sorter
// Values are encoded as 1->0 transitions on lines
// Sorts the inputs in ascending order of arrival times

`timescale 1ns / 1ps

module bitonic_sort_2 (input in1, input in2, output reg out1, output reg out2);
    
    always@(*)
    begin
    if (in1 >= in2) begin out2 = in1; out1 = in2; end
    if (in1 < in2)  begin out1 = in2; out2 = in1; end
    end 

    // Parameter declarations
    
    // Input/output declarations

    // Any temporary signal declarations

    // Instantiate smaller bitonic sorters

    // Last macro stage

endmodule
