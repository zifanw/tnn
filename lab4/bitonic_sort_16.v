// A 16-input bitonic sorter
// Values are encoded as 1->0 transitions on lines
// Sorts the inputs in ascending order of arrival times

`timescale 1ns / 1ps

module bitonic_sort_16 (output [15:0] sorted_out, input [15:0] raw_in);
   
   parameter N = 16;
   wire temp [0:4][0:N-1];
   wire [7:0] temp1;
   wire [7:0] temp2; 
   bitonic_sort_8 B34 (temp1, raw_in[7:0]);
   bitonic_sort_8 B35 (temp2, raw_in[15:8]);

   assign temp[0][0] = temp1[0];
   assign temp[0][1] = temp1[1];
   assign temp[0][2] = temp1[2];
   assign temp[0][3] = temp1[3];
   assign temp[0][4] = temp1[4];
   assign temp[0][5] = temp1[4];
   assign temp[0][6] = temp1[6];
   assign temp[0][7] = temp1[7];

   assign temp[0][8] = temp2[7];
   assign temp[0][9] = temp2[6];
   assign temp[0][10] = temp2[5];
   assign temp[0][11] = temp2[4];
   assign temp[0][12] = temp2[3];
   assign temp[0][13] = temp2[2];
   assign temp[0][14] = temp2[1];
   assign temp[0][15] = temp2[0];


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
 
   assign sorted_out[0] = temp[4][0];
   assign sorted_out[1] = temp[4][1];
   assign sorted_out[2] = temp[4][2];
   assign sorted_out[3] = temp[4][3];
   assign sorted_out[4] = temp[4][4];
   assign sorted_out[5] = temp[4][5];
   assign sorted_out[6] = temp[4][6];
   assign sorted_out[7] = temp[4][7];
   assign sorted_out[8] = temp[4][8];
   assign sorted_out[9] = temp[4][9];
   assign sorted_out[10] = temp[4][10];
   assign sorted_out[11] = temp[4][11];
   assign sorted_out[12] = temp[4][12];
   assign sorted_out[13] = temp[4][13];
   assign sorted_out[14] = temp[4][14];
   assign sorted_out[15] = temp[4][15];

   // Parameter declarations
    
    // Input/output declarations

    // Any temporary signal declarations

    // Instantiate smaller bitonic sorters

    // Last macro stage

endmodule
