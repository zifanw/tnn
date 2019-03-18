// A 16-input bitonic sorter
// Values are encoded as 1->0 transitions on lines
// Sorts the inputs in ascending order of arrival times

`timescale 1ns / 1ps

module bitonic_sort_8 (input in1, input in2, 
                       input in3, input in4,
                       input in5, input in6,
                       input in7, input in8,
                       output reg out1, output reg out2,
		       output reg out3, output reg out4,
                       output reg out5, output reg out6,
                       output reg out7, output reg out8);
    
   wire temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8;
   wire temp9, temp10, temp11, temp12, temp13, temp14, temp15, temp16;
 
   assign temp1 = in1;
   assign temp2 = in2;
   assign temp3 = in3;
   assign temp4 = in4;
   assign temp5 = in5;
   assign temp6 = in6;
   assign temp7 = in7;
   assign temp8 = in8;
   
   genvar i;
   generate
     for (i = 1; i < 4; i = i + 1) 
     begin:
 	bitonic_sort_4 B14 (temp1, temp2, temp3, temp4, temp9, temp10, temp11, temp12);
        bitonic_sort_4 B24 (temp5, temp6, temp7, temp8, temp13, temp14, temp15, temp16);
        assign temp1 = temp9;
        assign temp2 = temp13;
        assign temp3 = temp10;
        assign temp4 = temp14;
        assign temp5 = temp11;
        assign temp6 = temp15;
        assign temp7 = temp12;
        assign temp8 = temp16;
     end

   endgenerate
   // STAGE 1 
   //bitonic_sort_2 C1A    ( in1 ,in2 ,data_B1 ,data_S1);
   //bitonic_sort_2 C1B    ( in3 ,in4 ,data_B2 ,data_S2);


   // STAGE 2
   //bitonic_sort_2 C2A    ( data_B1 ,data_S2 ,data_B3 ,data_S3);  
   //bitonic_sort_2 C2B    ( data_S1 ,data_B2 ,data_B4 ,data_S4);  

   // STAGE 3
   //bitonic_sort_2 C3A    ( data_B3 ,data_S4 ,data_B5  ,data_S5 );
   //bitonic_sort_2 C3B    ( data_S3 ,data_B4 ,data_B6  ,data_S6 );



   assign out1 = temp1;
   assign out2 = temp2;
   assign out3 = temp3;
   assign out4 = temp4;
   assign out5 = temp5;
   assign out6 = temp6;
   assign out7 = temp7;
   assign out8 = temp8;
    // Parameter declarations
    
    // Input/output declarations

    // Any temporary signal declarations

    // Instantiate smaller bitonic sorters

    // Last macro stage

endmodule
