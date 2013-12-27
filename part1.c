#include <emmintrin.h>
#include <string.h>
#include <stdio.h>
#define KERNX 3 //this is the x-size of the kernel. It will always be odd.
#define KERNY 3 //this is the y-size of the kernel. It will always be odd.

/*
 *
 * Print Output Matrix in Grid Format
 *
 */
void print_grid(float* out, int data_size_X, int data_size_Y) {
    printf("-----NEW ARRAY-----\n");
    for(int y = 0; y < data_size_Y; ++y) 
    {
	    printf("|");
	    for(int x = 0; x < data_size_X; ++x)
	        printf(" %+2.2f | ", out[x+y*data_size_X]);
	    printf("\n");
	}
}

/*
 *
 * Generates new zero-padded array
 *
 */
float* pad_array(float* in, int data_size_X, int data_size_Y, int padx, int pady)
{
    //Allocate memory for padded array
    float *out = malloc((data_size_X+padx)*(data_size_Y+pady)*sizeof(float));
    int padx2 = padx/2;
    int pady2 = pady/2;
    //Zero entire array
    memset(out, 0, (data_size_X+padx)*(data_size_Y+pady)*sizeof(float));

    for(int i = 0; i < data_size_Y; i++) { 
        //Copy values in between padding
        memcpy(out+(padx2+(i+pady2)*(data_size_X+padx)), in+i*(data_size_X), (data_size_X) * sizeof(float)); 
    }

    return out;
}

/*
 *
 * Convolution is convoluted!
 *
 */
int conv2D(float* in, float* out, int data_size_X, int data_size_Y, float* kernel)
{
    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    
    //Padding calculations
    int padx = KERNX-1;
    int pady = KERNY-1;
    int padx2 = padx/2;
    int pady2 = pady/2;
    
    //Pad array
    float* in_p = pad_array(in, data_size_X, data_size_Y, padx, pady);
    
    // Main convolution loop
    
    // Loop ordering is fully optimized! (y, x, j, i)
    
    // Loop over Y
    for(int y = 0; y < data_size_Y; y++) 
    {
        // Unrolled Vectorized Loop over X
        for(int x = 0; x < data_size_X-15; x += 16) 
        {
            for(int j = -kern_cent_Y; j <= kern_cent_Y; j++)
            {
		        for(int i = -kern_cent_X; i <= kern_cent_X; i++)
		        {
                    __m128 k_tmp = _mm_load1_ps(kernel+(kern_cent_X-i)+(kern_cent_Y-j)*KERNX);
                    
			        __m128 in_tmp = _mm_loadu_ps(in_p+((0+x+padx2+i) + (y+pady2+j)*(data_size_X+padx)));
			        __m128 out_tmp = _mm_loadu_ps(out+(0+x+y*data_size_X));
			        _mm_storeu_ps(out+(0+x+y*data_size_X), _mm_add_ps(out_tmp, _mm_mul_ps(in_tmp, k_tmp)));
			
			        in_tmp = _mm_loadu_ps(in_p+((4+x+padx2+i) + (y+pady2+j)*(data_size_X+padx)));
			        out_tmp = _mm_loadu_ps(out+(4+x+y*data_size_X));
			        _mm_storeu_ps(out+(4+x+y*data_size_X), _mm_add_ps(out_tmp, _mm_mul_ps(in_tmp, k_tmp)));
			
			        in_tmp = _mm_loadu_ps(in_p+((8+x+padx2+i) + (y+pady2+j)*(data_size_X+padx)));
			        out_tmp = _mm_loadu_ps(out+(8+x+y*data_size_X));
			        _mm_storeu_ps(out+(8+x+y*data_size_X), _mm_add_ps(out_tmp, _mm_mul_ps(in_tmp, k_tmp)));
			
			        in_tmp = _mm_loadu_ps(in_p+((12+x+padx2+i) + (y+pady2+j)*(data_size_X+padx)));
			        out_tmp = _mm_loadu_ps(out+(12+x+y*data_size_X));
			        _mm_storeu_ps(out+(12+x+y*data_size_X), _mm_add_ps(out_tmp, _mm_mul_ps(in_tmp, k_tmp)));
			    }
			}
		}
		
		// Vectorized Loop over X
		for(int x = data_size_X - (data_size_X % 16); x < data_size_X-3; x += 4) 
		{
		    for(int j = -kern_cent_Y; j <= kern_cent_Y; j++)
            {
		        for(int i = -kern_cent_X; i <= kern_cent_X; i++)
		        {
		            __m128 k_tmp = _mm_load1_ps(kernel+(kern_cent_X-i)+(kern_cent_Y-j)*KERNX);
		            
		            __m128 in_tmp = _mm_loadu_ps(in_p+((0+x+padx2+i) + (y+pady2+j)*(data_size_X+padx)));
			        __m128 out_tmp = _mm_loadu_ps(out+(0+x+y*data_size_X));
			        _mm_storeu_ps(out+(0+x+y*data_size_X), _mm_add_ps(out_tmp, _mm_mul_ps(in_tmp, k_tmp)));
			    }
			}
		}
		
		// Final Loop over X (Cleanup)
		for(int x = data_size_X - (data_size_X % 4); x < data_size_X; x++) 
		{
		    for(int j = -kern_cent_Y; j <= kern_cent_Y; j++)
            {
		        for(int i = -kern_cent_X; i <= kern_cent_X; i++)
		        {
			        out[x+y*data_size_X] += 
			            kernel[(kern_cent_X-i)+(kern_cent_Y-j)*KERNX] * in_p[(x+padx2+i) + (y+pady2+j)*(data_size_X+padx)];
			    }
			}
		}
	}
	
	
	free(in_p); // Can't forget to free memory!
	
	
	return 1;
}


