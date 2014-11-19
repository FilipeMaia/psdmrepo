#ifndef PYPSALG_CPP_NDARRAYCVT_H
#define PYPSALG_CPP_NDARRAYCVT_H

#include <pytools/make_pyshared.h>
#include <pytools/PyDataType.h>

#include <boost/python.hpp>

#include <ndarray/ndarray.h>
namespace psana_python {
 
  // Function to create all the converters
  void createConverters();
  
  // Converter for NDArray to Numpy for BOOST
  template <typename T, unsigned Rank> struct NDArrayToNumpy 
  {
    // Converts NDArray to Numpy 
    static PyObject* convert(ndarray<T,Rank> const& array);

    // Registers NDArray to Numpy converter with BOOST
    static void register_ndarray_to_numpy_cvt();    
  };
  
  
  // Converter for Numpy to NDArray for BOOST
  template <typename T, unsigned Rank> struct NumpyToNDArray
  {
    // Registers Numpy to NDArray converter with BOOST
    NumpyToNDArray& from_python();

    // Check incoing PYTHON object can be converted to NDArray
    static void* convertible(PyObject* obj);
    
    // Converts Numpy to NDArray 
    typedef boost::python::converter::rvalue_from_python_stage1_data BoostData;
    static void  construct(PyObject* obj, BoostData* boostData);        
  };
  
}

#endif // PYPSALG_CPP_NDARRAYCVT_H
  
  


