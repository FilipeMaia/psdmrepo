#include <psalg/psalg.h>
#include <boost/python.hpp>
#include <ndarray/ndarray.h>
#include <algorithm>
#include "AreaDetHist.h"


// Wrappers for functions that use non-const references
// put in anonymous namespace for safety
namespace {
  // Wrapper for psalg::count_hits
  ndarray<unsigned,2> count_hits_1(const ndarray<const unsigned,2>&input,
				   unsigned threshold) {

    // Create output array...same size as input & initialized to zero
    ndarray<unsigned,2> output = make_ndarray<unsigned>(input.shape()[0], 
							input.shape()[1]);
    std::fill_n(output.begin(), output.size(), 0);

    // Call psalg::count_hits
    psalg::count_hits(input,threshold,output);

    // Return the output array
    return output;
  }


  ndarray<unsigned,2> count_hits_2(const ndarray<const unsigned,2>& input,
				   const ndarray<const unsigned,2>& threshold) {
    
    // Create output array...same size as input & initialized to zero
    ndarray<unsigned,2> output = make_ndarray<unsigned>(input.shape()[0], 
							input.shape()[1]);
    std::fill_n(output.begin(), output.size(), 0);
    
    // Call psalg::count_hits
    psalg::count_hits(input,threshold,output);

    // Return the output array
    return output;    
  }



  // Wrapper for psalg::sum_hits
  ndarray<unsigned,2> sum_hits_1(const ndarray<const unsigned,2>& input,
				 unsigned threshold, unsigned offset) {

    // Create output array...same size as input & initialized to zero
    ndarray<unsigned,2> output = make_ndarray<unsigned>(input.shape()[0], 
							input.shape()[1]);
    std::fill_n(output.begin(), output.size(), 0);
    
    // Call psalg::sum_hits
    psalg::sum_hits(input,threshold,offset,output);

    // Return the output array
    return output;   
  }


    ndarray<unsigned,2> sum_hits_2 (const ndarray<const unsigned,2>& input,
				  const ndarray<const unsigned,2>& threshold,
				  unsigned offset) {

    // Create output array...same size as input & initialized to zero
    ndarray<unsigned,2> output = make_ndarray<unsigned>(input.shape()[0], 
							input.shape()[1]);
    std::fill_n(output.begin(), output.size(), 0);

    // Call psalg::sum_hits
    psalg::sum_hits(input,threshold,offset,output);

    // Return the output array
    return output;        
  }



  // Wrapper for psalg::count_excess
  ndarray<unsigned,2> count_excess_1(const ndarray<const unsigned,2>& input ,
				     unsigned threshold) {
    
    // Create output array...same size as input & initialized to zero 
    ndarray<unsigned,2> output = make_ndarray<unsigned>(input.shape()[0], 
							input.shape()[1]);
    std::fill_n(output.begin(), output.size(), 0);
    
    // Call psalg::count_excess
    psalg::count_excess(input,threshold,output);

    // Return the output array
    return output;       
  }


  ndarray<unsigned,2> count_excess_2 (const ndarray<const unsigned,2>& input,
				      const ndarray<const unsigned,2>& threshold) {
    
    // Create output array...same size as input & initialized to zero
    ndarray<unsigned,2> output = make_ndarray<unsigned>(input.shape()[0], 
							input.shape()[1]);
    std::fill_n(output.begin(), output.size(), 0);
    
    // Call psalg::count_excess
    psalg::count_excess(input,threshold,output);

    // Return the output array
    return output;       
  }



  // Wrapper for psalg::sum_excess 
  ndarray<unsigned,2> sum_excess_1(const ndarray<const unsigned,2>& input,
				   unsigned threshold,
				   unsigned offset) {

    // Create output array...same size as input & initialized to zero
    ndarray<unsigned,2> output = make_ndarray<unsigned>(input.shape()[0], 
							input.shape()[1]);
    std::fill_n(output.begin(), output.size(), 0);
    
    // Call psalg::sum_excess
    psalg::sum_excess(input,threshold,offset,output);

    // Return the output array
    return output;       
  }


  ndarray<unsigned,2> sum_excess_2(const ndarray<const unsigned,2>& input,
				   const ndarray<const unsigned,2>& threshold,
				   unsigned offset) {

    // Create output array...same size as input & initialized to zero
    ndarray<unsigned,2> output = make_ndarray<unsigned>(input.shape()[0], 
							input.shape()[1]);
    std::fill_n(output.begin(), output.size(), 0);

    // Call psalg::sum_excess
    psalg::sum_excess(input,threshold,offset,output);

    // Return the output array
    return output;       
  }
  
  
  
  // Wrapper for rolling average
  ndarray<double,1> rolling_average_int32_t(const ndarray<const int32_t,1>& new_data,
					    const ndarray<const double,1>& old_data,
					    double fraction) {

    // Clone old_data
    ndarray<double,1> avg = old_data.copy();
    
    // Call psalg::rolling_average
    psalg::rolling_average<int32_t>(new_data,avg,fraction); 
    
    // Return the avg array
    return avg;     
  }


  ndarray<double,1> rolling_average_double(const ndarray<const double,1>& new_data,
					   const ndarray<const double,1>& old_data,
					   double fraction)  {

    // Clone old_data
    ndarray<double,1> avg = old_data.copy();
        
    // Call psalg::rolling_average
    psalg::rolling_average<double>(new_data,avg,fraction); 

    // Return the avg array
    return avg;         
  }  

}; //namespace


BOOST_PYTHON_MODULE(pypsalg_cpp)
{    

  // Create hooks between C++ psalg and PYTHON (via BOOST) NB: BOOST
  // (and PYTHON) require unqiue function pointers for each overloaded
  // function.  

  // NB: BOOST wrappers do not support functions with non-const
  // references. Those functions need a wrapper that has no non-const
  // references, or passes functions by value.

  // -- Finite Impulse response  
  ndarray<double,1> (*fimp)(const ndarray<const double,1>&,const ndarray<const double,1>&)
    = &psalg::finite_impulse_response;
  boost::python::def("finite_impulse_response",fimp,
		     "Finite impulse response filter \n"
		     "Creates the 1-dimensional filtered response array from the \n"
		     "sample input array and the impulse response filter array."
		     );
   
  
  // -- moments  
  ndarray<double,1>(*moments_1)(const ndarray<const double,1>&,double,double) = &psalg::moments;
  ndarray<double,1>(*moments_2)(const ndarray<const double,1>&,const ndarray<const double,1>&,
				double,double) = &psalg::moments;   
  ndarray<double,1>(*moments_3)(const ndarray<const unsigned,2>&,double) = &psalg::moments;
  ndarray<double,1>(*moments_4)(const ndarray<const unsigned,2>&,double,unsigned [][2]) = &psalg::moments;
  ndarray<double,1>(*moments_5)(const ndarray<const double,2>&,double) = &psalg::moments;
  ndarray<double,1>(*moments_6)(const ndarray<const double,2>&,double,unsigned [][2]) = &psalg::moments;
  ndarray<double,1>(*moments_7)(const ndarray<const unsigned,2>&,const ndarray<const unsigned,1>&,
  			       const ndarray<const unsigned,2>&,double) = &psalg::moments;
  ndarray<double,1>(*moments_8)(const ndarray<const unsigned,2>&,const ndarray<const unsigned,1>&,
  			       const ndarray<const unsigned,2>&,double,unsigned [][2]) = &psalg::moments;
  ndarray<double,1>(*moments_9)(const ndarray<const double,2>&,const ndarray<const unsigned,1>&,
  			       const ndarray<const unsigned,2>&,double) = &psalg::moments;
  ndarray<double,1>(*moments_10)(const ndarray<const double,2>&,const ndarray<const unsigned,1>&,
  				const ndarray<const unsigned,2>&,double,unsigned [][2]) = &psalg::moments;
  
  boost::python::def("moments_1D",moments_1,
		     "Calculates moments of 1D array. \n"
		     "Returns (sum of bin,\n"
		     "         sum of bin values*bin position, \n"
		     "         sum of bin values*bin position**2)\n\n"
		     "The bin_position is calculate as bin_offset + bin_index*bin_scale"
		     );
  boost::python::def("moments_1D",moments_2);  


  boost::python::def("moments_2D",moments_3,		     		     
		     "Calculate moments of 2-D array\n\n"   
		     "The moments are { sum of bins, \n"
		     "                  sum of bin_values,\n"
		     "                  sum of bin_values**2,\n"
		     "                  sum of bin_value*bin_xposition, \n"
		     "                  sum of bin_value*bin_yposition } \n\n"		     
		     "The bin_value is calculated as the array element value minus the \n"
		     "value_offset.  The bin_xposition(yposition) is simply the array index \n"
		     "for dimension 1(0).\n\n"		     
		     "Integral = moments[1] \n"
		     "Mean     = moments[1]/moments[0] \n"
		     "RMS      = sqrt((moments[2]/moments[0] - (moments[1]/moments[0])**2)\n"
		     "Contrast = sqrt(moments[0]*moments[2]/moments[1]**2 - 1) \n"
		     "X-center-of-mass = moments[3]/moments[1]\n"
		     "Y-center-of-mass = moments[4]/moments[1]\n"
		     );
  boost::python::def("moments_2D",moments_4);
  boost::python::def("moments_2D",moments_5);
  boost::python::def("moments_2D",moments_6);
  boost::python::def("moments_2D",moments_7);
  boost::python::def("moments_2D",moments_8);
  boost::python::def("moments_2D",moments_9);
  boost::python::def("moments_2D",moments_10);
  
  
  
  // -- Edge Finder
  boost::python::def("find_edges",&psalg::find_edges,
		     "EdgeFinder \n\n"		     
		     "Waveform pulse edge finder\n\n"   
		     "Generates an array of hit times and amplitudes for waveform\n"
		     "leading (trailing) edges using a constant fraction discriminator\n"
		     "algorithm.  The baseline and minimum amplitude threshold are used\n"
		     "for discriminating hits.  The pulse height fraction at which the hit\n"
		     "time is derived is also required as input.  Note that if the threshold\n"
		     "is less than the baseline value, then leading edges are 'falling' and \n"
		     "trailing edges are \"rising\".  In order for two pulses to be discriminated,\n"
		     "the waveform samples below the two pulses must fall below (or above for\n"
		     "negative pulses) the fractional value of the threshold; i.e. \n"
		     "waveform[i] < fraction*(threshold+baseline).\n\n"		     
		     "The results are stored in a 2D array such that result[i][0] is the time \n"
		     "(waveform sample) of the i'th hit and result[i][1] is the maximum amplitude \n"
		     "of the i'th hit."   
		     );

  
  // -- Hit Finder
  boost::python::def("count_hits",count_hits_1);
  boost::python::def("count_hits",count_hits_2,
		     "Image hit finder\n\n"  
		     "Generates a 2D map of hits, where a hit is defined as a local maximum above"
		     "some threshold.  The threshold can be a single value or a map of values.\n\n"   
		     "The results are stored in a 2D array with the same dimensions as the input image.\n\n"        
		     "Increment an output element when the input element is a local maximum and is\n"
		     "above threshold.  Threshold is either a constant or a map of threshold values."
		     );
  
  
  
  // -- Sum Hits
  boost::python::def("sum_hits",sum_hits_1);
  boost::python::def("sum_hits",sum_hits_2,		     
		     "Sum the input element's value into the output element when the input is a local\n"
		     "maximum and is above threshold.  The value of offset is subtracted from the\n"
		     "input value before adding to the output."  
		     );
  
  
  
  // -- Count Excess
  boost::python::def("count_excess",count_excess_1);
  boost::python::def("count_excess",count_excess_2,
		     "Increment output elements for all input elements above threshold.\n"
		     "The threshold can be a single value or a map of values."		     
		     );
  
  
  
  // -- Sum Excess
  boost::python::def("sum_excess",sum_excess_1);
  boost::python::def("sum_excess",sum_excess_2,
		     "Sum the input element's value into the output element when the input is\n"
		     "above threshold.  The value of offset is subtracted from the input value \n"
		     "before adding to the output."
		     );
  
  
  
  // -- Peak Fit
  double (*find_peak_1)(const ndarray<const double,1>&,double,const ndarray<const double,1>&,unsigned&) 
    = &psalg::find_peak;
  double (*find_peak_2)(const ndarray<const double,1>&,const ndarray<const double,1>&,
  			const ndarray<const double,1>&,unsigned&) = &psalg::find_peak;

  boost::python::def("find_peak",find_peak_1);
  boost::python::def("find_peak",find_peak_2);
  boost::python::def("find_peaks",&psalg::find_peaks,
		     "1D Peak find\n\n"   
		     "Find the peak value in the array.\n"
		     "Variable norm is number of entries summed into each bin."		     
		     );
  
  
  
  // -- Linear Fit
  ndarray<double,1> (*line_fit_1)(const ndarray<const double,1>&,const ndarray<const unsigned,1>&,double)
    = &psalg::line_fit;
  ndarray<double,1> (*line_fit_2)(const ndarray<const double,1>&,const ndarray<const unsigned,1>&,
  				  const ndarray<const double,1>&) = &psalg::line_fit;

  boost::python::def("line_fit",line_fit_1);
  boost::python::def("line_fit",line_fit_2,
		     "Variable norm is number of entries summed into each bin."
		     );
  
  
  
  // -- Dist RMS
  double (*dist_rms_1)(const ndarray<const double,1>&,double,const ndarray<const double,1>&) 
    = &psalg::dist_rms;
  double (*dist_rms_2)(const ndarray<const double,1>&,const ndarray<const double,1>&,
  		       const ndarray<const double,1>&) = &psalg::dist_rms;

  boost::python::def("dist_rms",dist_rms_2);
  boost::python::def("dist_rms",dist_rms_1,
		     "Distribution Root-mean-square\n\n"		     
		     "Width of distribution is estimated by the root-mean-square.\n"
		     "A baseline polynomial { f(i) = b[0] + i*b[1] + i*i*b[2] + ... }\n"
		     "is subtracted from each point prior to the rms calculation.\n"
		     "Points below the baseline contribute negatively to the rms."		     
		     );
  
  
  
  // -- Dist FWHM
  double (*dist_fwhm_1)(const ndarray<const double,1>&,double,const ndarray<const double,1>&) 
    = &psalg::dist_fwhm;
  double (*dist_fwhm_2)(const ndarray<const double,1>&,const ndarray<const double,1>&,
  			const ndarray<const double,1>&) = &psalg::dist_fwhm;

  boost::python::def("dist_fwhm",dist_fwhm_2);
  boost::python::def("dist_fwhm",dist_fwhm_1,
		     "Distribution Full-width-half-maximum\n\n"		     
		     "Width of distribution is estimated by the minimum full-width\n"
		     "half-maximum around the peak value."
		     );
  
  
  
  // -- Parabolic Interpolation
  ndarray<double,1> (*parab_interp_1)(const ndarray<const double,1>&,double,const ndarray<const double,1>&)
    = &psalg::parab_interp;
  ndarray<double,1> (*parab_interp_2)(const ndarray<const double,1>&,const ndarray<const double,1>&,
  				      const ndarray<const double,1>&) = &psalg::parab_interp;

  boost::python::def("parab_interp",parab_interp_2);
  boost::python::def("parab_interp",parab_interp_1,
		     "Parabolic interpolation\n\n"		     
		     "Perform a quadratic interpolation around the peak of the distribution.\n"
		     "A baseline polynomial { f(i) = b[0] + i*b[1] + i*i*b[2] + ... }\n"
		     "is subtracted from each point prior to the calculation.\n"
		     "Return value is an array of [ amplitude, position ]"
		     );
  
  
  
  // -- Parab Fit
  ndarray<double,1> (*parab_fit_1)(const ndarray<const double,1>&) = psalg::parab_fit;
  ndarray<double,1> (*parab_fit_2)(const ndarray<const double,1>&,unsigned,double) = psalg::parab_fit;

  boost::python::def("parab_fit", parab_fit_2);
  boost::python::def("parab_fit", parab_fit_1,
		     "Perform a least squares fit of the waveform to a 2nd-order polynomial.\n"
		     "Assumes all points have equal uncertainty.\n"
		     "Return value is an array of polynomial coefficients, such that \n"
		     "y(x) = a[0] + a[1]*x + a[2]*x**2\n"
		     "Maximum/minimum value is a[0]-a[1]*a[1]/(4*a[2]) at x=-a[1]/(2*a[2]).\n"
		     "Return array is [0,0,0] when fit fails."
		     );
  
  
  
  // -- Common mode 
  // These are difficult to use with BOOST as these functions pass raw C arrays
  // floats
  /*
  void (*commonMode_float_const)(const float*, const uint16_t*, const unsigned, const float, const float, float&)
    = &psalg::commonMode<float>;
  void (*commonMode_float)(float*, const uint16_t*, const unsigned, const float, const float, float& )
    = &psalg::commonMode<float>;
  void (*commonModeMedian_float_const)(const float*, const uint16_t*, const unsigned, const float, 
				       const float, float&) = &psalg::commonModeMedian<float>;  
  void (*commonModeMedian_float)(float*, const uint16_t*, const unsigned, const float, const float, float&)
    = &psalg::commonModeMedian<float>; 

  boost::python::def("common_mode",commonMode_float_const);
  boost::python::def("common_mode",commonMode_float);
  boost::python::def("common_mode_median",commonModeMedian_float_const);
  boost::python::def("common_mode_median",commonModeMedian_float);
  */

  // doubles
  /*
  void (*commonMode_double_const)(const double*, const uint16_t*, const unsigned, const double, 
				  const double, double&) = &psalg::commonMode<double>;
  void (*commonMode_double)(double*, const uint16_t*, const unsigned, const double, const double, double& )
    = &psalg::commonMode<double>;
  void (*commonModeMedian_double_const)(const double*, const uint16_t*, const unsigned, const double, 
					const double, double&) = &psalg::commonModeMedian<double>;  
  void (*commonModeMedian_double)(double*, const uint16_t*, const unsigned, const double, const double, double&)
    = &psalg::commonModeMedian<double>; 

  boost::python::def("common_mode",commonMode_double_const);
  boost::python::def("common_mode",commonMode_double);
  boost::python::def("common_mode_median",commonModeMedian_double_const);
  boost::python::def("common_mode_median",commonModeMedian_double);
  */

  boost::python::def("commonmode_lroe", &psalg::commonModeLROE,
		     "Calculate a common-mode in left-right halves for odd-even pixels");
  boost::python::def("project",&psalg::project,
		     "Project ndarray\n\n"   
		     "Creates a 1-dimensional response array from the"
		     "projection of an N-dimensional ndarray over a region of interest (inclusive).\n\n"		     
		     "pdim is the dimension to project onto."
		     "All other dimensions are integrated over the ROI"
		     );
  
  
  
  // -- ROLLING AVERAGE 
  boost::python::def("rolling_average",rolling_average_int32_t);
  boost::python::def("rolling_average",rolling_average_double,
		     "Accumulate a rolling average where each accumulation contributes\n"
		     "a fixed fraction to the average."
		     );
  
  static const char AreaDetClassDoc[] =
    "Class to manage histogramming of area detector ADU values";
  static const char AreaDetCtorDoc[] =
    "Arguments:\n"
    "- a 3 dimensional numpy array of doubles\n"
    "- int: lower limit of histogram\n"
    "- int: upper limit of histogram\n"
    "- bool: indicating whether to find isolated photons (default False)\n"
    "- double: threshold that the pixel should be above all neighbors\n"
    "  (valid only if the findIsolatedPhotons set to True)";
  static const char AreaDetUpdateDoc[] =
    "Arguments:\n"
    "- a 3 dimensional numpy array of doubles that will be used\n"
    "  to update histogram";
  static const char AreaDetGetDoc[] =
    "No arguments.  Return histogram as numpy array.";
  boost::python::class_<pypsalg::AreaDetHist>("AreaDetHist", AreaDetClassDoc, boost::python::init<ndarray<double,3>,int,int,bool,double>(AreaDetCtorDoc))
    .def("update",&pypsalg::AreaDetHist::update, AreaDetUpdateDoc)
    .def("get",&pypsalg::AreaDetHist::get, AreaDetGetDoc)
    ;
}
