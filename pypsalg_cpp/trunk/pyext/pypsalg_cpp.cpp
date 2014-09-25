#include <pyext/NdarrayCvt.h>
#include <pyext/numpytest.h>
#include <psalg/psalg.h>

#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/elem.hpp>


BOOST_PYTHON_MODULE(pypsalg_cpp)
{
  import_array();

  // set of ranks and types for which we instantiate converters
#define ND_RANKS (1)(2)(3)(4)(5)(6)
#define ND_TYPES (int8_t)(uint8_t)(int16_t)(uint16_t)(int32_t)(uint32_t)(int64_t)(uint64_t)(float)(double)
#define CONST_ND_TYPES (const int8_t)(const uint8_t)(const int16_t)(const uint16_t)(const int32_t)(const uint32_t)(const int64_t)(const uint64_t)(const float)(const double)
  

  // Preprocessor macro to define the call to create a NDArray to NUMPY converter. 
#define REGISTER_NDARRAY_TO_NUMPY_CONVERTER(r,PRODUCT) \
  boost::python::to_python_converter<ndarray<BOOST_PP_SEQ_ELEM(0,PRODUCT), \
					     BOOST_PP_SEQ_ELEM(1,PRODUCT)>,\
				     NDArrayToNumpy<BOOST_PP_SEQ_ELEM(0,PRODUCT),\
						    BOOST_PP_SEQ_ELEM(1,PRODUCT)> >(); 

  // Preprocessor macro to define the call to create a NUMPY to NDArray converter 
#define REGISTER_NUMPY_TO_NDARRAY_CONVERTER(r,PRODUCT) \
  NumpyToNDArray<BOOST_PP_SEQ_ELEM(0,PRODUCT),BOOST_PP_SEQ_ELEM(1,PRODUCT)>().from_python();
  
  // BOOST preprocessor macro to create converters 1-6 dimensional NDArray, with
  // data type from int to double, both const and non const  
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(REGISTER_NDARRAY_TO_NUMPY_CONVERTER,(ND_TYPES)(ND_RANKS))
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(REGISTER_NDARRAY_TO_NUMPY_CONVERTER,(CONST_ND_TYPES)(ND_RANKS))

  BOOST_PP_SEQ_FOR_EACH_PRODUCT(REGISTER_NUMPY_TO_NDARRAY_CONVERTER,(ND_TYPES)(ND_RANKS))
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(REGISTER_NUMPY_TO_NDARRAY_CONVERTER,(CONST_ND_TYPES)(ND_RANKS))

  // These templates and BOOST preprocessor macros can be hard to understand what's
  // going on.  Here's a couple of examples of what they trying to do....
  //
  // Let's say we want converters for doubles NDarrays from 1-3 dimensions   
  // Without the BOOST macros above we would have to do this:
  // 
  // boost::python::to_python_converter<ndarray<double,1>,NDArrayToNumpy<double,1> >();
  // boost::python::to_python_converter<ndarray<double,2>,NDArrayToNumpy<double,2> >();
  // boost::python::to_python_converter<ndarray<double,3>,NDArrayToNumpy<double,3> >();
  //  
  // NumpyToNDArray<float,1>().from_python();
  // NumpyToNDArray<float,2>().from_python();
  // NumpyToNDArray<float,3>().from_python();
  // 
  // Rather than typing each out each coverter, we used the BOOST
  // preprosser macros to generate the code for us.


  // Create hooks between C++ psalg and PYTHON (via BOOST)      
  // NB: For all overloaded functions in psalg, have to unique
  // function pointers to each function.  Otherwise the BOOST binding
  // wont know which function to bind with.

  // -- Finite Impulse response  
  ndarray<double,1> (*fimp1)(const ndarray<const double,1>&,const ndarray<const double,1>&)
    = &psalg::finite_impulse_response;
  void (*fimp2)(const ndarray<const double,1>&,const ndarray<const double,1>&,ndarray<double,1>&)
    = &psalg::finite_impulse_response;
  
  def("finite_impulse_response",fimp1);
  def("finite_impulse_response",fimp2);


  // -- Variance Accumulate
  void (*var_accum_1)(const ndarray<const double,2>&,ndarray<double,2>&,ndarray<double,2>&) 
    = &psalg::variance_accumulate;
  void (*var_accum_2)(double,const ndarray<const double,2>&,ndarray<double,2>&,ndarray<double,2>&)
    = &psalg::variance_accumulate;
  void (*var_accum_3)(double,const ndarray<const unsigned,2>&,ndarray<double,2>&,ndarray<double,2>&)
    = &psalg::variance_accumulate;
  void (*var_accum_4)(double,double,const ndarray<const unsigned,2>&,ndarray<double,2>&,ndarray<double,2>&)
    = &psalg::variance_accumulate;

  def("variance_accumulate",var_accum_1);  
  def("variance_accumulate",var_accum_2);  
  def("variance_accumulate",var_accum_3);  
  def("variance_accumulate",var_accum_4);  

  
  // -- Variance Calculate
  void (*var_calc_1)(double,const ndarray<const double,2>&,ndarray<double,2>&,ndarray<double,2>&,
  		     unsigned,ndarray<double,2>&) = &psalg::variance_calculate;
  void (*var_calc_2)(double,double,const ndarray<const unsigned,2>&,ndarray<double,2>&,ndarray<double,2>&,
  		     unsigned,ndarray<unsigned,2>&) = &psalg::variance_calculate;

  def("variance_calculate",var_calc_1);
  def("variance_calculate",var_calc_2);
  
  
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

  def("moments",moments_1);
  def("moments",moments_2);
  def("moments",moments_3);
  def("moments",moments_4);
  def("moments",moments_5);
  def("moments",moments_6);
  def("moments",moments_7);
  def("moments",moments_8);
  def("moments",moments_9);
  def("moments",moments_10);
  
  
  // -- Edge Finder
  def("find_edges",&psalg::find_edges);
  
  
  // -- Hit Finder
  void (*count_hits_1)(const ndarray<const unsigned,2>&,unsigned,ndarray<unsigned,2>&) 
    = &psalg::count_hits ;
  void (*count_hits_2)(const ndarray<const unsigned,2>&,const ndarray<const unsigned,2>&,
		       ndarray<unsigned,2>&) = &psalg::count_hits;

  def("count_hits",count_hits_1);
  def("count_hits",count_hits_2);
  
  
  // -- Sum Hits
  void (*sum_hits_1)(const ndarray<const unsigned,2>&,unsigned,unsigned,ndarray<unsigned,2>&) 
    = &psalg::sum_hits;
  void (*sum_hits_2)(const ndarray<const unsigned,2>&,const ndarray<const unsigned,2>&,unsigned,
  		     ndarray<unsigned,2>&) = &psalg::sum_hits;

  def("sum_hits",sum_hits_1);
  def("sum_hits",sum_hits_2);
  
  
  // -- Count Excess
  void (*count_excess_1)(const ndarray<const unsigned,2>&,unsigned,ndarray<unsigned,2>&) 
    = &psalg::count_hits;
  void (*count_excess_2)(const ndarray<const unsigned,2>&,const ndarray<const unsigned,2>&,
  			 ndarray<unsigned,2>&) = &psalg::count_hits;

  def("count_excess",count_excess_1);
  def("count_excess",count_excess_2);
  
  
  // -- Sum Excess
  void (*sum_excess_1)(const ndarray<const unsigned,2>&,unsigned,unsigned,ndarray<unsigned,2>&) 
    = &psalg::sum_excess;
  void (*sum_excess_2)(const ndarray<const unsigned,2>&,const ndarray<const unsigned,2>&,unsigned,
  		       ndarray<unsigned,2>&) = &psalg::sum_excess;

  def("sum_excess",sum_excess_1);
  def("sum_excess",sum_excess_2);
  
  
  // -- Peak Fit
  double (*find_peak_1)(const ndarray<const double,1>&,double,const ndarray<const double,1>&,unsigned&) 
    = &psalg::find_peak;
  double (*find_peak_2)(const ndarray<const double,1>&,const ndarray<const double,1>&,
  			const ndarray<const double,1>&,unsigned&) = &psalg::find_peak;

  def("find_peak",find_peak_1);
  def("find_peak",find_peak_2);
  def("find_peaks",&psalg::find_peaks);
  

  // -- Linear Fit
  ndarray<double,1> (*line_fit_1)(const ndarray<const double,1>&,const ndarray<const unsigned,1>&,double)
    = &psalg::line_fit;
  ndarray<double,1> (*line_fit_2)(const ndarray<const double,1>&,const ndarray<const unsigned,1>&,
  				  const ndarray<const double,1>&) = &psalg::line_fit;

  def("line_fit",line_fit_1);
  def("line_fit",line_fit_2);


  // -- Dist RMS
  double (*dist_rms_1)(const ndarray<const double,1>&,double,const ndarray<const double,1>&) 
    = &psalg::dist_rms;
  double (*dist_rms_2)(const ndarray<const double,1>&,const ndarray<const double,1>&,
  		       const ndarray<const double,1>&) = &psalg::dist_rms;

  def("dist_rms",dist_rms_1);
  def("dist_rms",dist_rms_2);
  
  
  // -- Dist FWHM
  double (*dist_fwhm_1)(const ndarray<const double,1>&,double,const ndarray<const double,1>&) 
    = &psalg::dist_fwhm;
  double (*dist_fwhm_2)(const ndarray<const double,1>&,const ndarray<const double,1>&,
  			const ndarray<const double,1>&) = &psalg::dist_fwhm;

  def("dist_fwhm",dist_fwhm_1);
  def("dist_fwhm",dist_fwhm_2);


  // -- Parabolic Interpolation
  ndarray<double,1> (*parab_interp_1)(const ndarray<const double,1>&,double,const ndarray<const double,1>&)
    = &psalg::parab_interp;
  ndarray<double,1> (*parab_interp_2)(const ndarray<const double,1>&,const ndarray<const double,1>&,
  				      const ndarray<const double,1>&) = &psalg::parab_interp;

  def("parab_interp",parab_interp_1);
  def("parab_interp",parab_interp_2);


  // -- Parab Fit
  ndarray<double,1> (*parab_fit_1)(const ndarray<const double,1>&) = psalg::parab_fit;
  ndarray<double,1> (*parab_fit_2)(const ndarray<const double,1>&,unsigned,double) = psalg::parab_fit;

  def("parab_fit", parab_fit_1);
  def("parab_fit", parab_fit_2);
  
  
  // -- Common mode 
  // floats
  void (*commonMode_float_const)(const float*, const uint16_t*, const unsigned, const float, const float, float&)
    = &psalg::commonMode<float>;
  void (*commonMode_float)(float*, const uint16_t*, const unsigned, const float, const float, float& )
    = &psalg::commonMode<float>;
  void (*commonModeMedian_float_const)(const float*, const uint16_t*, const unsigned, const float, 
				       const float, float&) = &psalg::commonModeMedian<float>;  
  void (*commonModeMedian_float)(float*, const uint16_t*, const unsigned, const float, const float, float&)
    = &psalg::commonModeMedian<float>; 

  def("common_mode",commonMode_float_const);
  def("common_mode",commonMode_float);
  def("common_mode_median",commonModeMedian_float_const);
  def("common_mode_median",commonModeMedian_float);
  
  // doubles
  void (*commonMode_double_const)(const double*, const uint16_t*, const unsigned, const double, 
				  const double, double&) = &psalg::commonMode<double>;
  void (*commonMode_double)(double*, const uint16_t*, const unsigned, const double, const double, double& )
    = &psalg::commonMode<double>;
  void (*commonModeMedian_double_const)(const double*, const uint16_t*, const unsigned, const double, 
					const double, double&) = &psalg::commonModeMedian<double>;  
  void (*commonModeMedian_double)(double*, const uint16_t*, const unsigned, const double, const double, double&)
    = &psalg::commonModeMedian<double>; 

  def("common_mode",commonMode_double_const);
  def("common_mode",commonMode_double);
  def("common_mode_median",commonModeMedian_double_const);
  def("common_mode_median",commonModeMedian_double);
  
  def("commonmode_lroe", &psalg::commonModeLROE);
  def("project",&psalg::project);
  

  // -- ROLLING AVERGAGE 
  void (*rolling_average_int32_t)(const ndarray<const int32_t,1>&, ndarray<double,1>&,double) 
    = &psalg::rolling_average<int32_t>;
  void (*rolling_average_double)(const ndarray<const double,1>&, ndarray<double,1>&,double) 
    = &psalg::rolling_average<double>;
  
  def("rolling_average",rolling_average_int32_t);
  def("rolling_average",rolling_average_double);

}
