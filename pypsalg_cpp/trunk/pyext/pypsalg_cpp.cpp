
#include <pyext/NdarrayCvt.h>
#include <pyext/numpytest.h>
#include <psalg/psalg.h>
#include <boost/ref.hpp>


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

  // Register Converters to BOOST
  boost::python::to_python_converter<ndarray<float,1>,NDArrayToNumpy<float,1> >();

  // Register Converters to BOOST
  boost::python::to_python_converter<ndarray<float,2>,NDArrayToNumpy<float,2> >();

  boost::python::to_python_converter<ndarray<double,1>,NDArrayToNumpy<double,1> >();

  boost::python::to_python_converter<ndarray<double,2>,NDArrayToNumpy<double,2> >();

  NumpyToNDArray<float,1>()
    .from_python()
    ;

  NumpyToNDArray<float,2>()
    .from_python()
    ;


  NumpyToNDArray<const double,1>()  
    .from_python()
    ;

  NumpyToNDArray<double,1>()
    .from_python()
    ;


  NumpyToNDArray<double,2>()
    .from_python()
    ;




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
