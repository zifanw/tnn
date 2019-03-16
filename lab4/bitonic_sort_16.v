// A 16-input bitonic sorter
// Values are encoded as 1->0 transitions on lines
// Sorts the inputs in ascending order of arrival times

`timescale 1ns / 1ps

module bitonic_sort_16 (sorted_out, raw_in)
   genvar i;
   generate
     for (i = 1; i < 4; i = i + 1) begin:
 	always@(*) begin 	
   		reg temp1 = raw_in[15];
   		reg temp2 = raw_in[14];
   		reg temp3 = raw_in[13];
  		reg temp4 = raw_in[12];
                reg temp5 = raw_in[11];
                reg temp6 = raw_in[10];
                reg temp7 = raw_in[9];
                reg temp8 = raw_in[8];
		reg temp9 = raw_in[7];
   		reg temp10 = raw_in[6];
   		reg temp11 = raw_in[5];
  		reg temp12 = raw_in[4];
                reg temp13 = raw_in[3];
                reg temp14 = raw_in[2];
                reg temp15 = raw_in[1];
                reg temp16 = raw_in[0];
                reg temp17, temp18, temp19, temp20, temp21, temp22, temp23, temp24;
                reg temp25, temp26, temp27, temp28, temp29, temp30, temp31, temp32;
 		bitonic_sort_4 B14 (temp1, temp2, temp3, temp4, temp5, temp6, temp7, temp8, 
				    temp17, temp18, temp19, temp20, temp21, temp22, temp23, temp24);
        	bitonic_sort_4 B24 (temp9, temp10, temp11, temp12, temp13, temp14, temp15, temp16,
				    temp25, temp26, temp27, temp28, temp29, temp30, temp31, temp32);
        	//TODO::I am still working on this logic
                temp1 = temp17;
        	temp2 = temp24;
        	temp3 = temp10;
        	temp4 = temp14;
                temp5 = temp11;
                temp6 = temp15;
                temp7 = temp12;
                temp8 = temp24;
        	temp9 = temp9;
        	temp10 = temp13;
        	temp11 = temp10;
        	temp12 = temp14;
                temp13 = temp11;
                temp14 = temp15;
                temp15 = temp12;
                temp16 = temp32;
		
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



   assign sorted_out[15] = temp1;
   assign sorted_out[14] = temp2;
   assign sorted_out[13] = temp3;
   assign sorted_out[12] = temp4;
   assign sorted_out[11] = temp5;
   assign sorted_out[10] = temp6;
   assign sorted_out[9] = temp7;
   assign sorted_out[8] = temp8;
   assign sorted_out[7] = temp9;
   assign sorted_out[6] = temp10;
   assign sorted_out[5] = temp11;
   assign sorted_out[4] = temp12;
   assign sorted_out[3] = temp13;
   assign sorted_out[2] = temp14;
   assign sorted_out[1] = temp15;
   assign sorted_out[0] = temp16;
    
    // Parameter declarations
    
    // Input/output declarations

    // Any temporary signal declarations

    // Instantiate smaller bitonic sorters

    // Last macro stage

endmodule
