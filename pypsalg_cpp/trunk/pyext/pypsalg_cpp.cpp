#include <pyext/NdarrayCvt.h>
#include <pyext/numpytest.h>
#include <psalg/psalg.h>

#include <boost/preprocessor/seq/for_each_product.hpp>
#include <boost/preprocessor/seq/elem.hpp>

//#include <boost/ref.hpp>
// ndarray<double,2> find_edges(const ndarray<const double,1>& wf,
// 			     double baseline_value,
// 			     double threshold_value,
// 			     double fraction,
// 			     double deadtime,
// 			     bool   leading_edge) {
//   numpytest numpyobj;
//   return  numpyobj.find_edges(wf,
// 			      baseline_value,
// 			      threshold_value,
// 			      fraction,
// 			      deadtime,
// 			      leading_edge);
// }






BOOST_PYTHON_MODULE(pypsalg_cpp)
{
  import_array();

  // set of ranks and types for which we instantiate converters
  // if you extend ND_TYPES macro also add specialization of Traits struct below
# define ND_CONST ( )(const)
#define ND_RANKS (1)(2)(3)(4)(5)(6)
#define ND_TYPES (int8_t)(uint8_t)(int16_t)(uint16_t)(int32_t)(uint32_t)(int64_t)(uint64_t)(float)(double)
#define CONST_ND_TYPES (const int8_t)(const uint8_t)(const int16_t)(const uint16_t)(const int32_t)(const uint32_t)(const int64_t)(const uint64_t)(const float)(const double)


#define REGISTER_CONVERTER(r,PRODUCT) \
  boost::python::to_python_converter<ndarray<BOOST_PP_SEQ_ELEM(0,PRODUCT),BOOST_PP_SEQ_ELEM(1,PRODUCT)>,NDArrayToNumpy<BOOST_PP_SEQ_ELEM(0,PRODUCT),BOOST_PP_SEQ_ELEM(1,PRODUCT)> >(); \
  NumpyToNDArray<BOOST_PP_SEQ_ELEM(0,PRODUCT),BOOST_PP_SEQ_ELEM(1,PRODUCT)>().from_python();
  
  BOOST_PP_SEQ_FOR_EACH_PRODUCT(REGISTER_CONVERTER,(ND_TYPES)(ND_RANKS))
    //  BOOST_PP_SEQ_FOR_EACH_PRODUCT(REGISTER_CONVERTER,(CONST_ND_TYPES)(ND_RANKS))

  // // Register Converters to BOOST
  // boost::python::to_python_converter<ndarray<float,1>,NDArrayToNumpy<float,1> >();

  // // Register Converters to BOOST
  // boost::python::to_python_converter<ndarray<float,2>,NDArrayToNumpy<float,2> >();

  // boost::python::to_python_converter<ndarray<double,1>,NDArrayToNumpy<double,1> >();

  // boost::python::to_python_converter<ndarray<double,2>,NDArrayToNumpy<double,2> >();

  // NumpyToNDArray<float,1>()
  //   .from_python()
  //   ;

  // NumpyToNDArray<float,2>()
  //   .from_python()
  //   ;


  // NumpyToNDArray<const double,1>()  
  //   .from_python()
  //   ;

  // NumpyToNDArray<double,1>()
  //   .from_python()
  //   ;


  // NumpyToNDArray<double,2>()
  //   .from_python()
  //   ;




  class_<numpytest>("numpytest")
    .def("printArray",&numpytest::printArray)
    .def("outArray",&numpytest::outArray)
    .def("calcmean", &numpytest::calcmean)
    .def("printArray2D",&numpytest::printArray2D)
    .def("find_edges",&numpytest::find_edges)
    ;

  //  def("find_edges",find_edges);
  

  class_<psalg>("psalg")
    .def("find_edges",&psalg::find_edges)
    ;
  
  // class_<psalg_ankush>("psalg_ankush")
  //   .def("find_edges",&psalg_ankush::find_edges)
  //   ;

  def("find_edges",&psalg::find_edges);


}
