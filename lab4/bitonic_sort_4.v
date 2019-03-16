// A 16-input bitonic sorter
// Values are encoded as 1->0 transitions on lines
// Sorts the inputs in ascending order of arrival times

`timescale 1ns / 1ps

module bitonic_sort_4 (input in1, input in2, 
                       input in3, input in4,
                       output reg out1, output reg out2,
		       output reg out3, output reg out4);
    
   genvar i;
   generate
     for (i = 0; i < 3; i = i + 1) begin:
 	always@(*) begin 	
   		reg temp1 = in1;
   		reg temp2 = in2;
   		reg temp3 = in3;
  		reg temp4 = in4;
                reg temp5, temp6, temp7, temp8;
 		bitonic_sort_2 B12 (temp1, temp2, temp5, temp6);
        	bitonic_sort_2 B22 (temp3, temp4, temp7, temp8);
        	temp1 = temp5;
        	temp2 = temp8;
        	temp3 = temp6;
        	temp4 = temp7;
        end
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
    // Parameter declarations
    
    // Input/output declarations

    // Any temporary signal declarations

    // Instantiate smaller bitonic sorters

    // Last macro stage

endmodule
