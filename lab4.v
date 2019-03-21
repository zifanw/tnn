`timescale 1ns / 1ps
module bitonic_sort_2 (in1, in2, out1, out2);  
    input in1, in2;
    output out1, out2;
    or U1 (out1, in1, in2);
    and U2 (out2, in1, in2); 
endmodule

module bitonic_sort_4 (output [3:0] sorted_out, input [3:0] raw_in);
   wire temp [0:2][0:3];
   bitonic_sort_2 B12 (raw_in[0], raw_in[1], temp[0][0], temp[0][1]);
   bitonic_sort_2 B22 (raw_in[2], raw_in[3], temp[0][3], temp[0][2]);
    
   genvar i;
   genvar j;
   genvar k;
   generate
   for (i = 1; i < $clog2(4)+1; i = i + 1)
     begin:substage
       for(j = 0; j < 4; j = j + 4/(2**(i-1)))
         begin:group
           for (k = j; k < j + 4/(2**i); k = k+1)
           begin:sort
           bitonic_sort_2 B23 (temp[i-1][k], temp[i-1][k+(4/(2**i))],temp[i][k], temp[i][k+(4/(2**i))]);
           end
         end
     end
   endgenerate
    
   assign sorted_out[0] = temp[2][0];
   assign sorted_out[1] = temp[2][1];  
   assign sorted_out[2] = temp[2][2];
   assign sorted_out[3] = temp[2][3];
endmodule

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


module bitonic_sort_16 (output [15:0] sorted_out, input [15:0] raw_in);
   parameter N = 16;
   wire temp [0:4][0:N-1];
   wire [7:0] temp1;
   wire [7:0] temp2; 
   bitonic_sort_8 B44 (temp1, raw_in[7:0]);
   bitonic_sort_8 B45 (temp2, raw_in[15:8]);

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
           bitonic_sort_2 B46 (temp[i-1][k], temp[i-1][k+N/(2**i)],temp[i][k], temp[i][k+N/(2**i)]);
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

endmodule
