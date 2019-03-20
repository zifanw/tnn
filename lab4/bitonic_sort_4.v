// A 16-input bitonic sorter
// Values are encoded as 1->0 transitions on lines
// Sorts the inputs in ascending order of arrival times

`timescale 1ns / 1ps

module bitonic_sort_4 (in1, in2, 
                       in3, in4,
                       out1, out2,
		       out3, out4);
   //wire temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
   
   input in1, in2, in3, in4;
   output out1, out2, out3, out4;      
   
   parameter N = 4; 
   wire [N-1:0] temp;
   bitonic_sort_2 B12 (in1, in2, temp[3], temp[2]);
   bitonic_sort_2 B22 (in3, in4, temp[1], temp[0]);
   

   wire temp_out1, temp_out2;
   genvar i;
   genvar j;
   genvar k;
   generate
   for (i = 1; i < $clog2(N)+1; i = i + 1)
     begin:substage
       for(j = 0; j < N; j = j + N/2**(i-1))
         begin:group
           for (k = j; 2*k < j + N; k = k+1)
           begin:sort
           bitonic_sort_2 B23 (temp[k], temp[k+(N/(2**i))], temp_out1, temp_out2);
           assign temp[k] = temp_out1;
           assign temp[k+(N/(2**i))] = temp_out2;
           end
         end
     end
   endgenerate

   assign out1 = temp[3];
   assign out2 = temp[2];
   assign out3 = temp[1];
   assign out4 = temp[0];
    // Parameter declarations
    
    // Input/output declarations

    // Any temporary signal declarations

    // Instantiate smaller bitonic sorters

    // Last macro stage

endmodule
