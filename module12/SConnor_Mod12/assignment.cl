//
// Book:      OpenCL(R) Programming Guide
// Authors:   Aaftab Munshi, Benedict Gaster, Dan Ginsburg, Timothy Mattson
// ISBN-10:   ??????????
// ISBN-13:   ?????????????
// Publisher: Addison-Wesley Professional
// URLs:      http://safari.informit.com/??????????
//            http://www.????????.com
//

// simple.cl
//
//    This is a simple example demonstrating buffers and sub-buffer usage

__kernel void square(__global float *input)
{
	size_t id = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t gid = get_group_id(0);

    if (lid == 0) {
        int sum = 0;
        for (int i = 0; i < 4; i++) {
            sum += input[id+i];
        }
        input[id] = sum/4.0;
    }
    else {
        input[id] = 0.0;
    }
}