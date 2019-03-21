// A 16-input bitonic sorter
// Values are encoded as 1->0 transitions on lines
// Sorts the inputs in ascending order of arrival times

`timescale 1ns / 1ps

module bitonic_sort_8 (output [7:0] sorted_out, input [7:0] raw_in);
 
   parameter N = 8; 
   wire temp [0:3][0:N-1];
   wire [3:0] temp1;
   wire [3:0] temp2; 
   bitonic_sort_4 B34 (temp1, raw_in[3:0]);
   bitonic_sort_4 B35 (temp2, raw_in[7:4]);
   assign temp[0][0] = temp1[0];
   assign temp[0][1] = temp1[1];
   assign temp[0][2] = temp1[2];
   assign temp[0][3] = temp1[3];
   assign temp[0][4] = temp2[3];
   assign temp[0][5] = temp2[2];
   assign temp[0][6] = temp2[1];
   assign temp[0][7] = temp2[0];
   
   genvar i;
   genvar j;
   genvar k;
   generate
   for (i = 1; i < $clog2(N)+1; i = i + 1)
     begin:substage
       for(j = 0; j < N; j = j + N/(2**(i-1)))
         begin:group
           for (k = j; k < j + N/(2**i); k = k+1)
           begin:sort

           bitonic_sort_2 B36 (temp[i-1][k], temp[i-1][k+N/(2**i)],temp[i][k], temp[i][k+N/(2**i)]);

           end
         end
     end
   endgenerate

   assign sorted_out[0] = temp[3][0];
   assign sorted_out[1] = temp[3][1];
   assign sorted_out[2] = temp[3][2];
   assign sorted_out[3] = temp[3][3];
   assign sorted_out[4] = temp[3][4];
   assign sorted_out[5] = temp[3][5];
   assign sorted_out[6] = temp[3][6];
   assign sorted_out[7] = temp[3][7];


endmodule
