
#include <pyext/NdarrayCvt.h>
#include <pyext/numpytest.h>

BOOST_PYTHON_MODULE(pypsalg_cpp)
{
  import_array();

  // Register Converters to BOOST
  boost::python::to_python_converter<ndarray<float,1>,NDArrayToNumpy<float,1> >();

  // Register Converters to BOOST
  boost::python::to_python_converter<ndarray<float,2>,NDArrayToNumpy<float,2> >();

  NumpyToNDArray<float,1>()
    .from_python()
    ;

  NumpyToNDArray<float,2>()
    .from_python()
    ;

  class_<numpytest>("numpytest")
    .def("printArray",&numpytest::printArray)
    .def("outArray",&numpytest::outArray)
    .def("calcmean", &numpytest::calcmean)
    ;

}
