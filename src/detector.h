#define DLIB_JPEG_SUPPORT

#include <dlib/svm_threaded.h>
#include <dlib/string.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>

#include <iostream>
#include <fstream>
#include <vector>

using namespace std;
using namespace dlib;


std::vector<rectangle> detect_objects(std::string imageFileName, std::string svmDetectorFileName) {
	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type; 

	// load a previously trained object detector and try it out on some data
	ifstream fin(svmDetectorFileName, ios::binary);
	if (!fin)
        throw new error("Cannot load svm detector file");

	object_detector<image_scanner_type> detector;
	deserialize(detector, fin);

	array2d<unsigned char> image;
    load_image(image, imageFileName);

    std::vector<rectangle> results = detector(image);
 
    return results;
}

