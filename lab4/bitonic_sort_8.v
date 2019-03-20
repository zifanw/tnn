// A 16-input bitonic sorter
// Values are encoded as 1->0 transitions on lines
// Sorts the inputs in ascending order of arrival times

`timescale 1ns / 1ps

module bitonic_sort_8 (in1, in2, 
                       in3, in4,
                       in5, in6,
                       in7, in8,
                       out1, out2,
		       out3, out4,
                       out5, out6,
                       out7, out8);
   //wire temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
   //wire temp9, temp10, temp11, temp12, temp13, temp14, temp15, temp16;
   input in1, in2, in3, in4, in5, in6, in7, in8;
   output out1, out2, out3, out4, out5, out6, out7, out8;
 
   parameter N = 8; 
   wire [N-1:0] temp;
   bitonic_sort_4 B14 (in1, in2, in3, in4, temp[7], temp[6], temp[5], temp[4]);
   bitonic_sort_4 B24 (in5, in6, in7, in8, temp[3], temp[2], temp[1], temp[0]);
   
   wire temp_out1, temp_out2;
   genvar i;
   genvar j;
   genvar k;
   generate
   for (i = 1; i < $clog2(N) + 1 ; i = i + 1)
     begin:substage
       for(j = 0; j < N; j = j + N/2**(i-1))
         begin:group
           for (k = j; 2*k < j+N; k = k+1)
           begin:sort
           bitonic_sort_2 B21 (temp[k], temp[k+(N/(2**i))], temp_out1, temp_out2);
           assign temp[k] = temp_out1;
           assign temp[k+(N/(2**i))] = temp_out2;
           end
         end
     end
   endgenerate
 
   assign out1 = temp[7];
   assign out2 = temp[6];
   assign out3 = temp[5];
   assign out4 = temp[4];
   assign out5 = temp[3];
   assign out6 = temp[2];
   assign out7 = temp[1];
   assign out8 = temp[0];
    // Parameter declarations
    
    // Input/output declarations

    // Any temporary signal declarations

    // Instantiate smaller bitonic sorters

    // Last macro stage

endmodule
