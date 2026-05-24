#include <iostream>
#include "functions.h"

cv::Mat ipa::faceRectangles(const cv::Mat & frame) 
{
	// find faces
	std::vector < cv::Rect > faces = ipa::faceDetector(frame);

	// for each face found...
	cv::Mat out = frame.clone();
	for(int k_x10=0; k_x10<faces.size(); k_x10++)
		cv::rectangle(out, faces[k_x10], cv::Scalar(0, 0, 255), 2);

	return out;
}