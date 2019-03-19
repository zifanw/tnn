// A 16-input bitonic sorter
// Values are encoded as 1->0 transitions on lines
// Sorts the inputs in ascending order of arrival times

`timescale 1ns / 1ps

module bitonic_sort_4 (in1, in2, 
                       in3, in4,
                       out1, out2,
		       out3, out4);
   wire temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
   
   input in1, in2, in3, in4;
   output out1, out2, out3, out4;      
   bitonic_sort_2 B12 (in1, in2, temp1, temp2);
   bitonic_sort_2 B22 (in3, in4, temp3, temp4);
         
   genvar i;
   generate
   for (i = 0; i < 2; i = i + 1) 
     begin:
 	bitonic_sort_2 B32 (temp1, temp3, temp5, temp6);
        bitonic_sort_2 B42 (temp2, temp4, temp7, temp8);
        assign temp1 = temp5;
        assign temp2 = temp8;
        assign temp3 = temp6;
        assign temp4 = temp7;
    end

   endgenerate

   assign out1 = temp1;
   assign out2 = temp2;
   assign out3 = temp3;
   assign out4 = temp4;
    // Parameter declarations
    
    // Input/output declarations

    // Any temporary signal declarations

    // Instantiate smaller bitonic sorters

    // Last macro stage

endmodule
