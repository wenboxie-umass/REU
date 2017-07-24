# REU

To the HLM_KMP_1D_Parallel.cpp file, g++-5 is required.

For Mac:
	g++-7 -O3 -Wall -Itrng-4.19 -Ltrng-4.19/src/.libs HLM_KMP_1D_Parallel.cpp -o exe_parallel -ltrng4 -fopenmp -std=c++11
	to compile the c++ file.