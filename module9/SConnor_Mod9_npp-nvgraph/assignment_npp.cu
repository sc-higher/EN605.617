#include <ImagesCPU.h>
#include <ImagesNPP.h>
#include <ImageIO.h>
#include <string.h>
#include <npp.h>

/* ========================================================================== */

int main(int argc, char** argv) {
	
    // set filename of image - must be in directory
    std::string sFilename = "dog.pgm";    
    // declare a single channel host image
    npp::ImageCPU_8u_C1 oHostSrc;
    // load image to host
    npp::loadImage(sFilename, oHostSrc);
    // copy to device
    npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);
    // create NppiSize struct 
    NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
    // allocate device and host output images of appropriate size
    npp::ImageNPP_8u_C1 oDeviceDst(oSrcSize.width, oSrcSize.height);
    npp::ImageCPU_8u_C1 oHostDst(oDeviceDst.size());

    // run mirror
    nppiMirror_8u_C1R (
        oDeviceSrc.data(),
        oDeviceSrc.pitch(),
        oDeviceDst.data(),
        oDeviceDst.pitch(),
        oSrcSize,
        NPP_VERTICAL_AXIS );       

    // copy the device result data into host image
    oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());

    // set filename of result image and save
    std::string sResultFilename = "dog_mirror.pgm";
    saveImage(sResultFilename, oHostDst);
    
    return 0;
	
} /* END main() */

/* ========================================================================== */
