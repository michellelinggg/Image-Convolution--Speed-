#include <emmintrin.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>
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
    int padx2 = padx/2;
    int pady2 = pady/2;

    //Allocate memory for padded array
    float *out = malloc((data_size_X+padx)*(data_size_Y+pady)*sizeof(float));
    
    //Register block a zero
    __m128 z = _mm_setzero_ps();
    
    //Zero top
    #pragma omp parallel for
    for(int j = 0; j < pady2; j++) 
    {
        for(int i = 0; i < (data_size_X+padx)-3; i += 4)
            _mm_storeu_ps(out+(i+j*(data_size_X+padx)), z);
            
        for(int i = (data_size_X+padx) - ((data_size_X+padx) % 4); i < data_size_X+padx; i++)
            out[i + j*(data_size_X+padx)] = 0;       
    }

    #pragma omp parallel for
    for(int j = 0; j < data_size_Y; j++) { 
        //Copy values in between padding
        for(int i = 0; i < data_size_X-3; i += 4)
            _mm_storeu_ps(out+(i+padx2+(j+pady2)*(data_size_X+padx)), _mm_loadu_ps(in+(i+j*data_size_X)));
            
        for(int i = data_size_X - (data_size_X % 4); i < data_size_X; i++)
            out[i+padx2+(j+pady2)*(data_size_X+padx)] = in[i+j*data_size_X];
        
        //Zero left/right
        for(int i = 0; i < padx2; i++) {
            out[(j+padx2)*(data_size_X + padx)+i] = 0;
            out[(j+padx2)*(data_size_X + padx)+i+(data_size_X + padx2)] = 0; 
        }
    }
    
    //Zero bottom
    #pragma omp parallel for
    for(int j = data_size_Y+pady2; j < data_size_Y+pady; j++) 
    {
        for(int i = 0; i < (data_size_X+padx)-3; i += 4)
            _mm_storeu_ps(out+(i+j*(data_size_X+padx)), z);
            
        for(int i = (data_size_X+padx) - ((data_size_X+padx) % 4); i < data_size_X+padx; i++)
            out[i + j*(data_size_X+padx)] = 0;       
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
    // register declarations
    __m128 sum0, sum4, sum8, sum12, sum16, sum20, sum24, sum28, k_tmp, in_tmp;

    // the x coordinate of the kernel's center
    int kern_cent_X = (KERNX - 1)/2;
    
    // the y coordinate of the kernel's center
    int kern_cent_Y = (KERNY - 1)/2;
    
    //Padding calculations
    int padx = KERNX-1;
    int pady = KERNY-1;
    int padx2 = padx/2;
    int pady2 = pady/2;
    int tmp_yx;
    
    //Pad array
    float* in_p = pad_array(in, data_size_X, data_size_Y, padx, pady);
    
    // Main convolution loop
    
    // Loop ordering is fully optimized! (y, x, j, i)
    
    // Loop over Y
    #pragma omp parallel for private(sum0, sum4, sum8, sum12, sum16, sum20, sum24, sum28, k_tmp, in_tmp, tmp_yx)
    for(int y = 0; y < data_size_Y; y++) 
    {
        // Unrolled Vectorized Loop over X
        for(int x = 0; x < data_size_X-31; x += 32) 
        {
            // Only load values once
            sum0  = _mm_loadu_ps(out+(0+x+y*data_size_X));
            sum4  = _mm_loadu_ps(out+(4+x+y*data_size_X));
            sum8  = _mm_loadu_ps(out+(8+x+y*data_size_X));
            sum12 = _mm_loadu_ps(out+(12+x+y*data_size_X));
            sum16 = _mm_loadu_ps(out+(16+x+y*data_size_X));
            sum20 = _mm_loadu_ps(out+(20+x+y*data_size_X));
            sum24 = _mm_loadu_ps(out+(24+x+y*data_size_X));
            sum28 = _mm_loadu_ps(out+(28+x+y*data_size_X));
            
            for(int j = -kern_cent_Y; j <= kern_cent_Y; j++)
            {
		        for(int i = -kern_cent_X; i <= kern_cent_X; i++)
		        {
		            tmp_yx = x+padx2+i+(y+pady2+j)*(data_size_X+padx);
		            
                    k_tmp = _mm_load1_ps(kernel+(kern_cent_X-i)+(kern_cent_Y-j)*KERNX);
                    
			        in_tmp = _mm_loadu_ps(in_p+((0+tmp_yx)));
			        sum0 = _mm_add_ps(sum0, _mm_mul_ps(in_tmp, k_tmp));
			        
			        in_tmp = _mm_loadu_ps(in_p+((4+tmp_yx)));
			        sum4 = _mm_add_ps(sum4, _mm_mul_ps(in_tmp, k_tmp));
			        
			        in_tmp = _mm_loadu_ps(in_p+((8+tmp_yx)));
			        sum8 = _mm_add_ps(sum8, _mm_mul_ps(in_tmp, k_tmp));
			        
			        in_tmp = _mm_loadu_ps(in_p+((12+tmp_yx)));
			        sum12 = _mm_add_ps(sum12, _mm_mul_ps(in_tmp, k_tmp));
			        
			        in_tmp = _mm_loadu_ps(in_p+((16+tmp_yx)));
			        sum16 = _mm_add_ps(sum16, _mm_mul_ps(in_tmp, k_tmp));
			        
			        in_tmp = _mm_loadu_ps(in_p+((20+tmp_yx)));
			        sum20 = _mm_add_ps(sum20, _mm_mul_ps(in_tmp, k_tmp));
			        
			        in_tmp = _mm_loadu_ps(in_p+((24+tmp_yx)));
			        sum24 = _mm_add_ps(sum24, _mm_mul_ps(in_tmp, k_tmp));
			        
			        in_tmp = _mm_loadu_ps(in_p+((28+tmp_yx)));
			        sum28 = _mm_add_ps(sum28, _mm_mul_ps(in_tmp, k_tmp));
			    }
			}
			
			// Only store values once
			_mm_storeu_ps(out+(0+x+y*data_size_X), sum0);
			_mm_storeu_ps(out+(4+x+y*data_size_X), sum4);
			_mm_storeu_ps(out+(8+x+y*data_size_X), sum8);
			_mm_storeu_ps(out+(12+x+y*data_size_X), sum12);
			_mm_storeu_ps(out+(16+x+y*data_size_X), sum16);
			_mm_storeu_ps(out+(20+x+y*data_size_X), sum20);
			_mm_storeu_ps(out+(24+x+y*data_size_X), sum24);
			_mm_storeu_ps(out+(28+x+y*data_size_X), sum28);
		}
		
		for(int x = data_size_X - (data_size_X % 32); x < data_size_X-15; x += 16) 
        {
            // Only load values once
            sum0  = _mm_loadu_ps(out+(0+x+y*data_size_X));
            sum4  = _mm_loadu_ps(out+(4+x+y*data_size_X));
            sum8  = _mm_loadu_ps(out+(8+x+y*data_size_X));
            sum12 = _mm_loadu_ps(out+(12+x+y*data_size_X));
            
            for(int j = -kern_cent_Y; j <= kern_cent_Y; j++)
            {
		        for(int i = -kern_cent_X; i <= kern_cent_X; i++)
		        {
		            tmp_yx = x+padx2+i+(y+pady2+j)*(data_size_X+padx);
		            
                    k_tmp = _mm_load1_ps(kernel+(kern_cent_X-i)+(kern_cent_Y-j)*KERNX);
                    
			        in_tmp = _mm_loadu_ps(in_p+((0+tmp_yx)));
			        sum0 = _mm_add_ps(sum0, _mm_mul_ps(in_tmp, k_tmp));
			        
			        in_tmp = _mm_loadu_ps(in_p+((4+tmp_yx)));
			        sum4 = _mm_add_ps(sum4, _mm_mul_ps(in_tmp, k_tmp));
			        
			        in_tmp = _mm_loadu_ps(in_p+((8+tmp_yx)));
			        sum8 = _mm_add_ps(sum8, _mm_mul_ps(in_tmp, k_tmp));
			        
			        in_tmp = _mm_loadu_ps(in_p+((12+tmp_yx)));
			        sum12 = _mm_add_ps(sum12, _mm_mul_ps(in_tmp, k_tmp));
			    }
			}
			
			// Only store values once
			_mm_storeu_ps(out+(0+x+y*data_size_X), sum0);
			_mm_storeu_ps(out+(4+x+y*data_size_X), sum4);
			_mm_storeu_ps(out+(8+x+y*data_size_X), sum8);
			_mm_storeu_ps(out+(12+x+y*data_size_X), sum12);
		}
		
		// Vectorized Loop over X
		for(int x = data_size_X - (data_size_X % 16); x < data_size_X-7; x += 8) 
		{
		    sum0 = _mm_loadu_ps(out+(0+x+y*data_size_X));
		    sum4 = _mm_loadu_ps(out+(4+x+y*data_size_X));
		    
		    for(int j = -kern_cent_Y; j <= kern_cent_Y; j++)
            {
		        for(int i = -kern_cent_X; i <= kern_cent_X; i++)
		        {
		            tmp_yx = x+padx2+i+(y+pady2+j)*(data_size_X+padx);
		        
		            k_tmp = _mm_load1_ps(kernel+(kern_cent_X-i)+(kern_cent_Y-j)*KERNX);
		            in_tmp = _mm_loadu_ps(in_p+((0+tmp_yx)));
			        sum0 = _mm_add_ps(sum0, _mm_mul_ps(in_tmp, k_tmp));
			        
			        in_tmp = _mm_loadu_ps(in_p+((4+tmp_yx)));
			        sum4 = _mm_add_ps(sum4, _mm_mul_ps(in_tmp, k_tmp));
			    }
			}
			_mm_storeu_ps(out+(0+x+y*data_size_X), sum0);
			_mm_storeu_ps(out+(4+x+y*data_size_X), sum4);
		}
		
		// Vectorized Loop over X
		for(int x = data_size_X - (data_size_X % 8); x < data_size_X-3; x += 4) 
		{
		    sum0 = _mm_loadu_ps(out+(0+x+y*data_size_X));
		    for(int j = -kern_cent_Y; j <= kern_cent_Y; j++)
            {
		        for(int i = -kern_cent_X; i <= kern_cent_X; i++)
		        {
		            k_tmp = _mm_load1_ps(kernel+(kern_cent_X-i)+(kern_cent_Y-j)*KERNX);
		            in_tmp = _mm_loadu_ps(in_p+((0+x+padx2+i) + (y+pady2+j)*(data_size_X+padx)));
			        sum0 = _mm_add_ps(sum0, _mm_mul_ps(in_tmp, k_tmp));
			    }
			}
			_mm_storeu_ps(out+(0+x+y*data_size_X), sum0);
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



