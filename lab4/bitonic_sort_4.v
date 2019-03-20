// A 16-input bitonic sorter
// Values are encoded as 1->0 transitions on lines
// Sorts the inputs in ascending order of arrival times

`timescale 1ns / 1ps

module bitonic_sort_4 (output [3:0] sorted_out, input [3:0] raw_in);

   //wire temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
       
   parameter N = 4; 
   wire [N-1:0] temp [2:0];
   bitonic_sort_2 B12 (raw_in[0], raw_in[1], temp[0][0], temp[0][1]);
   bitonic_sort_2 B22 (raw_in[2], raw_in[3], temp[0][2], temp[0][3]);
   

   //wire temp_out1, temp_out2;
   //wire [N-1:0] temp_out;
  // wire [1:0] temp_out;
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
	   //com1 = {temp[k], temp[k+(N/(2**i))]};
	   //temp_out = {temp_out1, temp_out2};
           bitonic_sort_2 B23 (temp[i-1][k], temp[i-1][k+(N/(2**i))],temp[i][k], temp[i][k+(N/(2**i))]);
	   //{temp_out1, temp_out2} = temp_out;
           //assign temp[k] = temp_out1;
           //assign temp[k+(N/(2**i))] = temp_out2;
           end
         end
     end
   endgenerate

   assign sorted_out[0] = temp[2][0];
   assign sorted_out[1] = temp[2][1];
   assign sorted_out[2] = temp[2][2];
   assign sorted_out[3] = temp[2][3];
    // Parameter declarations
    
    // Input/output declarations

    // Any temporary signal declarations

    // Instantiate smaller bitonic sorters

    // Last macro stage

endmodule
