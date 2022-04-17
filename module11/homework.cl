//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Timothy Mattson, James Fung, Dan Ginsburg
// ISBN-10:   0-321-74964-2
// ISBN-13:   978-0-321-74964-2
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/9780132488006/
//            http://www.openclprogrammingguide.com
//

// Convolution.cl
//
//    This is a simple kernel performing convolution.

__kernel void convolve(
	const __global  float * const input,
    __constant float * const mask,
    __global  float * const output,
    const int inputWidth,
    const int maskWidth)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    float sum = 0.0;
    uint distance = 0;
    float gradient = 0.0;
    const uint center = (maskWidth - 1) / 2;
    int r_dist = 0;
    int c_dist = 0;

    for (int r = 0; r < maskWidth; r++)
    {
        const int idxIntmp = (y + r) * inputWidth + x;
        r_dist = r-center;

        for (int c = 0; c < maskWidth; c++)
        {
			c_dist = c-center;
            distance = ( abs(r_dist) >= abs(c_dist) ? abs(r_dist) : abs(c_dist) );

            if (distance == 0) {
                gradient = 1.00;
            } else if (distance == 1) {
                gradient = 0.75;
            } else if (distance == 2) {
                gradient = 0.50;
            } else {
                gradient = 0.25;
            }            
            
            sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c] * gradient;
            // sum += mask[(r * maskWidth)  + c] * input[idxIntmp + c];
        }
    } 
    
	output[y * get_global_size(0) + x] = sum;
}