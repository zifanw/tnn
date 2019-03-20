// A 16-input bitonic sorter
// Values are encoded as 1->0 transitions on lines
// Sorts the inputs in ascending order of arrival times

`timescale 1ns / 1ps

module bitonic_sort_16 (output [15:0] sorted_out, input [15:0] raw_in);
   
   parameter N = 16;
   wire [N-1:0] temp;
   bitonic_sort_8 B18 (raw_in[15], raw_in[14], raw_in[13], raw_in[12], raw_in[11], raw_in[10], raw_in[9], raw_in[8], 
			temp[15], temp[14], temp[13], temp[12], temp[11], temp[10], temp[9], temp[8]);
   bitonic_sort_8 B28 (raw_in[7], raw_in[6], raw_in[5], raw_in[4], raw_in[3], raw_in[2], raw_in[1], raw_in[0], 
			temp[7], temp[6], temp[5], temp[4], temp[3], temp[2], temp[1], temp[0]);
   
   wire temp_out1, temp_out2;
   genvar i;
   genvar j;
   genvar k;
   generate
   for (i = 1; i < $clog2(N) + 1; i = i + 1)
     begin:substage
       for(j = 0; j < N; j = j + N/2**(i-1))
         begin:group
           for (k = j; 2*k < j + N; k = k+1)
           begin:sort
           bitonic_sort_2 B21 (temp[k], temp[k+(N/(2**i))], temp_out1, temp_out2);
           assign temp[k] = temp_out1;
           assign temp[k+(N/(2**i))] = temp_out2;
           end
         end
     end
   endgenerate
 
   assign sorted_out = temp;
 
    // Parameter declarations
    
    // Input/output declarations

    // Any temporary signal declarations

    // Instantiate smaller bitonic sorters

    // Last macro stage

endmodule
