
#include <boost/python.hpp>
#include "ndarray/ndarray.h"
//#include <cstddef>  // for size_t

#include "ImgAlgos/AlgArrProc.h"

//-------------------
using namespace ImgAlgos;

typedef ImgAlgos::AlgArrProc::mask_t mask_t;   // uint16_t
typedef ImgAlgos::AlgArrProc::wind_t wind_t;   // uint32_t

//-------------------

void     (AlgArrProc::*p_swin1) (ndarray<const wind_t,2>) = &AlgArrProc::setWindows;
void     (AlgArrProc::*p_sson1) (const float&, const float&) = &AlgArrProc::setSoNPars;
void     (AlgArrProc::*p_spsp1) (const float&, const float&, const float&, const float&, const float&) = &AlgArrProc::setPeakSelectionPars;

unsigned (AlgArrProc::*p_npix_f2) (ndarray<const float,   2>, ndarray<const mask_t,2>, const float&)    = &AlgArrProc::numberOfPixAboveThr<float,   2>;
unsigned (AlgArrProc::*p_npix_d2) (ndarray<const double,  2>, ndarray<const mask_t,2>, const double&)   = &AlgArrProc::numberOfPixAboveThr<double,  2>;
unsigned (AlgArrProc::*p_npix_i2) (ndarray<const int,     2>, ndarray<const mask_t,2>, const int&)      = &AlgArrProc::numberOfPixAboveThr<int,     2>;
unsigned (AlgArrProc::*p_npix_s2) (ndarray<const int16_t, 2>, ndarray<const mask_t,2>, const int16_t&)  = &AlgArrProc::numberOfPixAboveThr<int16_t, 2>;
unsigned (AlgArrProc::*p_npix_u2) (ndarray<const uint16_t,2>, ndarray<const mask_t,2>, const uint16_t&) = &AlgArrProc::numberOfPixAboveThr<uint16_t,2>;

unsigned (AlgArrProc::*p_npix_f3) (ndarray<const float,   3>, ndarray<const mask_t,3>, const float&)    = &AlgArrProc::numberOfPixAboveThr<float,   3>;
unsigned (AlgArrProc::*p_npix_d3) (ndarray<const double,  3>, ndarray<const mask_t,3>, const double&)   = &AlgArrProc::numberOfPixAboveThr<double,  3>;
unsigned (AlgArrProc::*p_npix_i3) (ndarray<const int,     3>, ndarray<const mask_t,3>, const int&)      = &AlgArrProc::numberOfPixAboveThr<int,     3>;
unsigned (AlgArrProc::*p_npix_s3) (ndarray<const int16_t, 3>, ndarray<const mask_t,3>, const int16_t&)  = &AlgArrProc::numberOfPixAboveThr<int16_t, 3>;
unsigned (AlgArrProc::*p_npix_u3) (ndarray<const uint16_t,3>, ndarray<const mask_t,3>, const uint16_t&) = &AlgArrProc::numberOfPixAboveThr<uint16_t,3>;

//-------------------

double (AlgArrProc::*p_ipix_f2) (ndarray<const float,   2>, ndarray<const mask_t,2>, const float&)    = &AlgArrProc::intensityOfPixAboveThr<float,   2>;
double (AlgArrProc::*p_ipix_d2) (ndarray<const double,  2>, ndarray<const mask_t,2>, const double&)   = &AlgArrProc::intensityOfPixAboveThr<double,  2>;
double (AlgArrProc::*p_ipix_i2) (ndarray<const int,     2>, ndarray<const mask_t,2>, const int&)      = &AlgArrProc::intensityOfPixAboveThr<int,     2>;
double (AlgArrProc::*p_ipix_s2) (ndarray<const int16_t, 2>, ndarray<const mask_t,2>, const int16_t&)  = &AlgArrProc::intensityOfPixAboveThr<int16_t, 2>;
double (AlgArrProc::*p_ipix_u2) (ndarray<const uint16_t,2>, ndarray<const mask_t,2>, const uint16_t&) = &AlgArrProc::intensityOfPixAboveThr<uint16_t,2>;

double (AlgArrProc::*p_ipix_f3) (ndarray<const float,   3>, ndarray<const mask_t,3>, const float&)    = &AlgArrProc::intensityOfPixAboveThr<float,   3>;
double (AlgArrProc::*p_ipix_d3) (ndarray<const double,  3>, ndarray<const mask_t,3>, const double&)   = &AlgArrProc::intensityOfPixAboveThr<double,  3>;
double (AlgArrProc::*p_ipix_i3) (ndarray<const int,     3>, ndarray<const mask_t,3>, const int&)      = &AlgArrProc::intensityOfPixAboveThr<int,     3>;
double (AlgArrProc::*p_ipix_s3) (ndarray<const int16_t, 3>, ndarray<const mask_t,3>, const int16_t&)  = &AlgArrProc::intensityOfPixAboveThr<int16_t, 3>;
double (AlgArrProc::*p_ipix_u3) (ndarray<const uint16_t,3>, ndarray<const mask_t,3>, const uint16_t&) = &AlgArrProc::intensityOfPixAboveThr<uint16_t,3>;

//-------------------

ndarray<const float, 2> (AlgArrProc::*p_pfv01_f2) (ndarray<const float,   2>, ndarray<const mask_t,2>, const float&,    const float&,    const unsigned&, const float&) = &AlgArrProc::dropletFinder<float,   2>;
ndarray<const float, 2> (AlgArrProc::*p_pfv01_d2) (ndarray<const double,  2>, ndarray<const mask_t,2>, const double&,   const double&,   const unsigned&, const float&) = &AlgArrProc::dropletFinder<double,  2>;
ndarray<const float, 2> (AlgArrProc::*p_pfv01_i2) (ndarray<const int,     2>, ndarray<const mask_t,2>, const int&,      const int&,      const unsigned&, const float&) = &AlgArrProc::dropletFinder<int,     2>;
ndarray<const float, 2> (AlgArrProc::*p_pfv01_s2) (ndarray<const int16_t, 2>, ndarray<const mask_t,2>, const int16_t&,  const int16_t&,  const unsigned&, const float&) = &AlgArrProc::dropletFinder<int16_t, 2>;
ndarray<const float, 2> (AlgArrProc::*p_pfv01_u2) (ndarray<const uint16_t,2>, ndarray<const mask_t,2>, const uint16_t&, const uint16_t&, const unsigned&, const float&) = &AlgArrProc::dropletFinder<uint16_t,2>;
																			              
ndarray<const float, 2> (AlgArrProc::*p_pfv01_f3) (ndarray<const float,   3>, ndarray<const mask_t,3>, const float&,    const float&,    const unsigned&, const float&) = &AlgArrProc::dropletFinder<float,   3>;
ndarray<const float, 2> (AlgArrProc::*p_pfv01_d3) (ndarray<const double,  3>, ndarray<const mask_t,3>, const double&,   const double&,   const unsigned&, const float&) = &AlgArrProc::dropletFinder<double,  3>;
ndarray<const float, 2> (AlgArrProc::*p_pfv01_i3) (ndarray<const int,     3>, ndarray<const mask_t,3>, const int&,      const int&,      const unsigned&, const float&) = &AlgArrProc::dropletFinder<int,     3>;
ndarray<const float, 2> (AlgArrProc::*p_pfv01_s3) (ndarray<const int16_t, 3>, ndarray<const mask_t,3>, const int16_t&,  const int16_t&,  const unsigned&, const float&) = &AlgArrProc::dropletFinder<int16_t, 3>;
ndarray<const float, 2> (AlgArrProc::*p_pfv01_u3) (ndarray<const uint16_t,3>, ndarray<const mask_t,3>, const uint16_t&, const uint16_t&, const unsigned&, const float&) = &AlgArrProc::dropletFinder<uint16_t,3>;

//-------------------

ndarray<const float, 2> (AlgArrProc::*p_pfv02_f2) (ndarray<const float,   2>, ndarray<const mask_t,2>, const float&,    const float&, const float&) = &AlgArrProc::peakFinder<float,   2>;
ndarray<const float, 2> (AlgArrProc::*p_pfv02_d2) (ndarray<const double,  2>, ndarray<const mask_t,2>, const double&,   const float&, const float&) = &AlgArrProc::peakFinder<double,  2>;
ndarray<const float, 2> (AlgArrProc::*p_pfv02_i2) (ndarray<const int,     2>, ndarray<const mask_t,2>, const int&,      const float&, const float&) = &AlgArrProc::peakFinder<int,     2>;
ndarray<const float, 2> (AlgArrProc::*p_pfv02_s2) (ndarray<const int16_t, 2>, ndarray<const mask_t,2>, const int16_t&,  const float&, const float&) = &AlgArrProc::peakFinder<int16_t, 2>;
ndarray<const float, 2> (AlgArrProc::*p_pfv02_u2) (ndarray<const uint16_t,2>, ndarray<const mask_t,2>, const uint16_t&, const float&, const float&) = &AlgArrProc::peakFinder<uint16_t,2>;
																		  
ndarray<const float, 2> (AlgArrProc::*p_pfv02_f3) (ndarray<const float,   3>, ndarray<const mask_t,3>, const float&,    const float&, const float&) = &AlgArrProc::peakFinder<float,   3>;
ndarray<const float, 2> (AlgArrProc::*p_pfv02_d3) (ndarray<const double,  3>, ndarray<const mask_t,3>, const double&,   const float&, const float&) = &AlgArrProc::peakFinder<double,  3>;
ndarray<const float, 2> (AlgArrProc::*p_pfv02_i3) (ndarray<const int,     3>, ndarray<const mask_t,3>, const int&,      const float&, const float&) = &AlgArrProc::peakFinder<int,     3>;
ndarray<const float, 2> (AlgArrProc::*p_pfv02_s3) (ndarray<const int16_t, 3>, ndarray<const mask_t,3>, const int16_t&,  const float&, const float&) = &AlgArrProc::peakFinder<int16_t, 3>;
ndarray<const float, 2> (AlgArrProc::*p_pfv02_u3) (ndarray<const uint16_t,3>, ndarray<const mask_t,3>, const uint16_t&, const float&, const float&) = &AlgArrProc::peakFinder<uint16_t,3>;

//-------------------
//void (AlgArrProc::*p_set01) (const float&, const float&) = &AlgArrProc::setSoNPars;
//void (ImgAlgos::AlgArrProc::*print_1) () = &ImgAlgos::AlgArrProc::printInputPars;
//-------------------

// BOOST wrapper to create imgalgos_ext module that contains the ImgAlgos::AlgArrProc
// python class that calls the C++ ImgAlgos::AlgArrProc methods
// NB: The name of the python module (imgalgos_ext) MUST match the name given in
// PYEXTMOD in the SConscript

BOOST_PYTHON_MODULE(imgalgos_ext)
{    
  using namespace boost::python;

  //boost::python::class_<AlgArrProc>("AlgArrProc", init<const unsigned&>())
  boost::python::class_<AlgArrProc>("AlgArrProc", init<ndarray<const wind_t,2>, const unsigned&>())
 
    .def("set_windows", p_swin1)     
    .def("set_peak_selection_pars", p_spsp1)     
    .def("set_son_pars", p_sson1)     
    .def("print_input_pars", &AlgArrProc::printInputPars)

    //.def("set_son_parameters", p_set01)     

    .def("number_of_pix_above_thr_f2", p_npix_f2)
    .def("number_of_pix_above_thr_d2", p_npix_d2)
    .def("number_of_pix_above_thr_i2", p_npix_i2)
    .def("number_of_pix_above_thr_s2", p_npix_s2)
    .def("number_of_pix_above_thr_u2", p_npix_u2)
    				    	  	    
    .def("number_of_pix_above_thr_f3", p_npix_f3)
    .def("number_of_pix_above_thr_d3", p_npix_d3)
    .def("number_of_pix_above_thr_i3", p_npix_i3)
    .def("number_of_pix_above_thr_s3", p_npix_s3)
    .def("number_of_pix_above_thr_u3", p_npix_u3)


    .def("intensity_of_pix_above_thr_f2", p_ipix_f2)
    .def("intensity_of_pix_above_thr_d2", p_ipix_d2)
    .def("intensity_of_pix_above_thr_i2", p_ipix_i2)
    .def("intensity_of_pix_above_thr_s2", p_ipix_s2)
    .def("intensity_of_pix_above_thr_u2", p_ipix_u2)
    	  		    	  	    
    .def("intensity_of_pix_above_thr_f3", p_ipix_f3)
    .def("intensity_of_pix_above_thr_d3", p_ipix_d3)
    .def("intensity_of_pix_above_thr_i3", p_ipix_i3)
    .def("intensity_of_pix_above_thr_s3", p_ipix_s3)
    .def("intensity_of_pix_above_thr_u3", p_ipix_u3)


    .def("peak_finder_v1_f2", p_pfv01_f2)
    .def("peak_finder_v1_d2", p_pfv01_d2)
    .def("peak_finder_v1_i2", p_pfv01_i2)
    .def("peak_finder_v1_s2", p_pfv01_s2)
    .def("peak_finder_v1_u2", p_pfv01_u2)
    	  
    .def("peak_finder_v1_f3", p_pfv01_f3)
    .def("peak_finder_v1_d3", p_pfv01_d3)
    .def("peak_finder_v1_i3", p_pfv01_i3)
    .def("peak_finder_v1_s3", p_pfv01_s3)
    .def("peak_finder_v1_u3", p_pfv01_u3)


    .def("peak_finder_v2_f2", p_pfv02_f2)
    .def("peak_finder_v2_d2", p_pfv02_d2)
    .def("peak_finder_v2_i2", p_pfv02_i2)
    .def("peak_finder_v2_s2", p_pfv02_s2)
    .def("peak_finder_v2_u2", p_pfv02_u2)
    	   
    .def("peak_finder_v2_f3", p_pfv02_f3)
    .def("peak_finder_v2_d3", p_pfv02_d3)
    .def("peak_finder_v2_i3", p_pfv02_i3)
    .def("peak_finder_v2_s3", p_pfv02_s3)
    .def("peak_finder_v2_u3", p_pfv02_u3)
  ;
}

//-------------------

