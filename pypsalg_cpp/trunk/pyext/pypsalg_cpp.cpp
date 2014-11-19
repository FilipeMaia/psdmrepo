#include <pypsalg_cpp/NdarrayCvt.h>
//   #include <pypsalg_cpp/numpytest.h>

#include <psalg/psalg.h>

#include <boost/python.hpp>
#include <ndarray/ndarray.h>



BOOST_PYTHON_MODULE(pypsalg_cpp)
{    
  // Create converters
  psana_python::createConverters();


  // Create hooks between C++ psalg and PYTHON (via BOOST)      
  // NB: For all overloaded functions in psalg, have to unique
  // function pointers to each function.  Otherwise the BOOST binding
  // wont know which function to bind with.

  // -- Finite Impulse response  
  ndarray<double,1> (*fimp1)(const ndarray<const double,1>&,const ndarray<const double,1>&)
    = &psalg::finite_impulse_response;
  void (*fimp2)(const ndarray<const double,1>&,const ndarray<const double,1>&,ndarray<double,1>&)
    = &psalg::finite_impulse_response;
  
  boost::python::def("finite_impulse_response",fimp1);
  boost::python::def("finite_impulse_response",fimp2);


  // -- Variance Accumulate
  void (*var_accum_1)(const ndarray<const double,2>&,ndarray<double,2>&,ndarray<double,2>&) 
    = &psalg::variance_accumulate;
  void (*var_accum_2)(double,const ndarray<const double,2>&,ndarray<double,2>&,ndarray<double,2>&)
    = &psalg::variance_accumulate;
  void (*var_accum_3)(double,const ndarray<const unsigned,2>&,ndarray<double,2>&,ndarray<double,2>&)
    = &psalg::variance_accumulate;
  void (*var_accum_4)(double,double,const ndarray<const unsigned,2>&,ndarray<double,2>&,ndarray<double,2>&)
    = &psalg::variance_accumulate;

  boost::python::def("variance_accumulate",var_accum_1);  
  boost::python::def("variance_accumulate",var_accum_2);  
  boost::python::def("variance_accumulate",var_accum_3);  
  boost::python::def("variance_accumulate",var_accum_4);  

  
  // -- Variance Calculate
  void (*var_calc_1)(double,const ndarray<const double,2>&,ndarray<double,2>&,ndarray<double,2>&,
  		     unsigned,ndarray<double,2>&) = &psalg::variance_calculate;
  void (*var_calc_2)(double,double,const ndarray<const unsigned,2>&,ndarray<double,2>&,ndarray<double,2>&,
  		     unsigned,ndarray<unsigned,2>&) = &psalg::variance_calculate;

  boost::python::def("variance_calculate",var_calc_1);
  boost::python::def("variance_calculate",var_calc_2);
  
  
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

  boost::python::def("moments",moments_1);
  boost::python::def("moments",moments_2);
  boost::python::def("moments",moments_3);
  boost::python::def("moments",moments_4);
  boost::python::def("moments",moments_5);
  boost::python::def("moments",moments_6);
  boost::python::def("moments",moments_7);
  boost::python::def("moments",moments_8);
  boost::python::def("moments",moments_9);
  boost::python::def("moments",moments_10);
  
  
  // -- Edge Finder
  boost::python::def("find_edges",&psalg::find_edges);
  
  
  // -- Hit Finder
  void (*count_hits_1)(const ndarray<const unsigned,2>&,unsigned,ndarray<unsigned,2>&) 
    = &psalg::count_hits ;
  void (*count_hits_2)(const ndarray<const unsigned,2>&,const ndarray<const unsigned,2>&,
		       ndarray<unsigned,2>&) = &psalg::count_hits;

  boost::python::def("count_hits",count_hits_1);
  boost::python::def("count_hits",count_hits_2);
  
  
  // -- Sum Hits
  void (*sum_hits_1)(const ndarray<const unsigned,2>&,unsigned,unsigned,ndarray<unsigned,2>&) 
    = &psalg::sum_hits;
  void (*sum_hits_2)(const ndarray<const unsigned,2>&,const ndarray<const unsigned,2>&,unsigned,
  		     ndarray<unsigned,2>&) = &psalg::sum_hits;

  boost::python::def("sum_hits",sum_hits_1);
  boost::python::def("sum_hits",sum_hits_2);
  
  
  // -- Count Excess
  void (*count_excess_1)(const ndarray<const unsigned,2>&,unsigned,ndarray<unsigned,2>&) 
    = &psalg::count_hits;
  void (*count_excess_2)(const ndarray<const unsigned,2>&,const ndarray<const unsigned,2>&,
  			 ndarray<unsigned,2>&) = &psalg::count_hits;

  boost::python::def("count_excess",count_excess_1);
  boost::python::def("count_excess",count_excess_2);
  
  
  // -- Sum Excess
  void (*sum_excess_1)(const ndarray<const unsigned,2>&,unsigned,unsigned,ndarray<unsigned,2>&) 
    = &psalg::sum_excess;
  void (*sum_excess_2)(const ndarray<const unsigned,2>&,const ndarray<const unsigned,2>&,unsigned,
  		       ndarray<unsigned,2>&) = &psalg::sum_excess;

  boost::python::def("sum_excess",sum_excess_1);
  boost::python::def("sum_excess",sum_excess_2);
  
  
  // -- Peak Fit
  double (*find_peak_1)(const ndarray<const double,1>&,double,const ndarray<const double,1>&,unsigned&) 
    = &psalg::find_peak;
  double (*find_peak_2)(const ndarray<const double,1>&,const ndarray<const double,1>&,
  			const ndarray<const double,1>&,unsigned&) = &psalg::find_peak;

  boost::python::def("find_peak",find_peak_1);
  boost::python::def("find_peak",find_peak_2);
  boost::python::def("find_peaks",&psalg::find_peaks);
  

  // -- Linear Fit
  ndarray<double,1> (*line_fit_1)(const ndarray<const double,1>&,const ndarray<const unsigned,1>&,double)
    = &psalg::line_fit;
  ndarray<double,1> (*line_fit_2)(const ndarray<const double,1>&,const ndarray<const unsigned,1>&,
  				  const ndarray<const double,1>&) = &psalg::line_fit;

  boost::python::def("line_fit",line_fit_1);
  boost::python::def("line_fit",line_fit_2);


  // -- Dist RMS
  double (*dist_rms_1)(const ndarray<const double,1>&,double,const ndarray<const double,1>&) 
    = &psalg::dist_rms;
  double (*dist_rms_2)(const ndarray<const double,1>&,const ndarray<const double,1>&,
  		       const ndarray<const double,1>&) = &psalg::dist_rms;

  boost::python::def("dist_rms",dist_rms_1);
  boost::python::def("dist_rms",dist_rms_2);
  
  
  // -- Dist FWHM
  double (*dist_fwhm_1)(const ndarray<const double,1>&,double,const ndarray<const double,1>&) 
    = &psalg::dist_fwhm;
  double (*dist_fwhm_2)(const ndarray<const double,1>&,const ndarray<const double,1>&,
  			const ndarray<const double,1>&) = &psalg::dist_fwhm;

  boost::python::def("dist_fwhm",dist_fwhm_1);
  boost::python::def("dist_fwhm",dist_fwhm_2);


  // -- Parabolic Interpolation
  ndarray<double,1> (*parab_interp_1)(const ndarray<const double,1>&,double,const ndarray<const double,1>&)
    = &psalg::parab_interp;
  ndarray<double,1> (*parab_interp_2)(const ndarray<const double,1>&,const ndarray<const double,1>&,
  				      const ndarray<const double,1>&) = &psalg::parab_interp;

  boost::python::def("parab_interp",parab_interp_1);
  boost::python::def("parab_interp",parab_interp_2);


  // -- Parab Fit
  ndarray<double,1> (*parab_fit_1)(const ndarray<const double,1>&) = psalg::parab_fit;
  ndarray<double,1> (*parab_fit_2)(const ndarray<const double,1>&,unsigned,double) = psalg::parab_fit;

  boost::python::def("parab_fit", parab_fit_1);
  boost::python::def("parab_fit", parab_fit_2);
  
  
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

  boost::python::def("common_mode",commonMode_float_const);
  boost::python::def("common_mode",commonMode_float);
  boost::python::def("common_mode_median",commonModeMedian_float_const);
  boost::python::def("common_mode_median",commonModeMedian_float);
  
  // doubles
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
  
  boost::python::def("commonmode_lroe", &psalg::commonModeLROE);
  boost::python::def("project",&psalg::project);
  

  // -- ROLLING AVERGAGE 
  void (*rolling_average_int32_t)(const ndarray<const int32_t,1>&, ndarray<double,1>&,double) 
    = &psalg::rolling_average<int32_t>;
  void (*rolling_average_double)(const ndarray<const double,1>&, ndarray<double,1>&,double) 
    = &psalg::rolling_average<double>;
  
  boost::python::def("rolling_average",rolling_average_int32_t);
  boost::python::def("rolling_average",rolling_average_double);

  


}
