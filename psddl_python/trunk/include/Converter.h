#ifndef PSDDL_PYTHON_CONVERTER_H
#define PSDDL_PYTHON_CONVERTER_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id: PyDataType.h 5266 2013-01-31 20:14:36Z salnikov@SLAC.STANFORD.EDU $
//
// Description:
//      Class Converter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include "python/Python.h"
#include <typeinfo>
#include <vector>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "PSEvt/ProxyDictI.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//    ---------------------
//    -- Class Interface --
//    ---------------------

namespace psddl_python {

/// @addtogroup psddl_python

/**
 *  @ingroup psddl_python
 *
 *  @brief Class defining interface for "converter" object types.
 *
 *  Instances of converter types know how to convert data between
 *  C++ format and Python objects. One converter instance can convert
 *  C++ object to Python, or Python objects to C++, or both.
 *
 *  @see ConverterMap
 *
 *  This software was developed for the LCLS project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 */

class Converter {
public:

  virtual ~Converter() {}

  /**
   *  @brief Return type_infos of source C++ types.
   *
   *  If converter supports conversion from C++ to Python
   *  this method shall return non-empty vector.
   */
  virtual std::vector<const std::type_info*> from_cpp_types() const = 0;

  /**
   *  @brief Returns source Python types.
   *
   *  If converter supports conversion from Python to C++
   *  this method shall return non-empty vector.
   *
   *  @return Borrowed references
   */
  virtual std::vector<PyTypeObject*> from_py_types() const = 0;

  /**
   *  @brief Returns destination Python types.
   *
   *  If converter supports conversion from C++ to Python
   *  this method shall return non-empty vector.
   *
   *  @return Borrowed references
   */
  virtual std::vector<PyTypeObject*> to_py_types() const = 0;

  /**
   *  @brief Return list of pdsdata::TypeId::Type enum values.
   */
  virtual std::vector<int> from_pds_types() const { return std::vector<int>(); }

  /**
   *  @brief Convert C++ object to Python
   *
   *  This method may return zero pointer if this converter cannot do conversion.
   *
   *  @return New reference
   */
  virtual PyObject* convert(PSEvt::ProxyDictI& proxyDict, const PSEvt::Source& source, const std::string& key) const = 0;

  /**
   *  @brief Convert Python object to C++
   *
   *  @return True for successful conversion, false otherwise
   */
  virtual bool convert(PyObject* obj, PSEvt::ProxyDictI& proxyDict, const Pds::Src& source, const std::string& key) const
  {
    return false;
  }

};

}

#endif // PSDDL_PYTHON_CONVERTER_H
