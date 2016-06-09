/**
 * Modified version of the train_object_detector example (DLib's source code)
 */

#define DLIB_JPEG_SUPPORT

#include <dlib/svm_threaded.h>
#include <dlib/string.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/cmd_line_parser.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace dlib;

struct TrainingRecord {
    std::string imageFileName;
    std::vector<dlib::rectangle> matchAreas;
};

// Define the best window size based on the rectangles defined for the images
void pick_best_window_size (
    const std::vector<std::vector<rectangle> >& boxes,
    unsigned long& width,
    unsigned long& height,
    const unsigned long target_size
)
/*!
    ensures
        - Finds the average aspect ratio of the elements of boxes and outputs a width
          and height such that the aspect ratio is equal to the average and also the
          area is equal to target_size.  That is, the following will be approximately true:
            - #width*#height == target_size
            - #width/#height == the average aspect ratio of the elements of boxes.
!*/
{
    // find the average width and height
    running_stats<double> avg_width, avg_height;
    for (unsigned long i = 0; i < boxes.size(); ++i)
    {
        for (unsigned long j = 0; j < boxes[i].size(); ++j)
        {
            avg_width.add(boxes[i][j].width());
            avg_height.add(boxes[i][j].height());
        }
    }

    // now adjust the box size so that it is about target_pixels pixels in size
    double size = avg_width.mean()*avg_height.mean();
    double scale = std::sqrt(target_size/size);

    width = (unsigned long)(avg_width.mean()*scale+0.5);
    height = (unsigned long)(avg_height.mean()*scale+0.5);
    // make sure the width and height never round to zero.
    if (width == 0)
        width = 1;
    if (height == 0)
        height = 1;
}

bool contains_any_boxes (
    const std::vector<std::vector<rectangle> >& boxes
)
{
    for (unsigned long i = 0; i < boxes.size(); ++i)
    {
        if (boxes[i].size() != 0)
            return true;
    }
    return false;
}

void throw_invalid_box_error_message (
    const std::vector<std::vector<rectangle> >& removed,
    const unsigned long target_size
)
{
    std::ostringstream sout;
    sout << "Error!  An impossible set of object boxes was given for training. ";
    sout << "All the boxes need to have a similar aspect ratio and also not be ";
    sout << "smaller than about " << target_size << " pixels in area. ";
    throw error("\n"+wrap_string(sout.str()) + "\n");
}

//===================================================== Actual code comes now ===========
void train_object_detector(std::vector<TrainingRecord>& trainingRecords, std::string detectorOutputFileName) {
	typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type; 
	// Get the upsample option from the user but use 0 if it wasn't given.
	const unsigned long upsample_amount = 0;

	dlib::array<array2d<unsigned char> > images;
	std::vector<std::vector<rectangle> > object_locations, ignore;

    //ignore = load_image_dataset(images, object_locations, "/home/danp3d/personal/roadsigns/pattern-recognition/data/dataset.xml");

    ignore.clear(); // not used for now. Keep it simple.
    images.clear();
    object_locations.clear();
    
    std::vector<rectangle> empty;
    empty.clear();
    array2d<unsigned char> img;
    for (int i = 0; i < trainingRecords.size(); ++i) {
        TrainingRecord* rec = &trainingRecords[i];
        load_image(img, rec->imageFileName);
        images.push_back(img);
        object_locations.push_back(rec->matchAreas);
        ignore.push_back(empty);
    }

    // Default values
	const int threads = 4;
	const double C   = 1.0;
	const double eps = 0.01;
	unsigned int num_folds = 3;
	const unsigned long target_size = 80*80;
	// You can't do more folds than there are images.  
	if (num_folds > images.size())
		num_folds = images.size();

	//upsample_image_dataset<pyramid_down<2> >(images, object_locations, ignore);


	image_scanner_type scanner;
	unsigned long width, height;
	pick_best_window_size(object_locations, width, height, target_size);
	scanner.set_detection_window_size(width, height); 

	structural_object_detection_trainer<image_scanner_type> trainer(scanner);
	trainer.set_num_threads(threads);
	trainer.set_c(C);
	trainer.set_epsilon(eps);

	// Now make sure all the boxes are obtainable by the scanner.  
	std::vector<std::vector<rectangle> > removed;
	removed = remove_unobtainable_rectangles(trainer, images, object_locations);
	// if we weren't able to get all the boxes to match then throw an error 
	if (contains_any_boxes(removed))
	{
		unsigned long scale = upsample_amount+1;
		scale = scale*scale;
		throw_invalid_box_error_message(removed, target_size/scale);
	}

    // Do the actual training and save the results into the detector object.  
    object_detector<image_scanner_type> detector = trainer.train(images, object_locations, ignore);
    serialize(detectorOutputFileName) << detector;
}

