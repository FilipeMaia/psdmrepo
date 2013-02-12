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
 *  Instances of converter types know how to convert data from data from 
 *  C++ format into Python objects. There will be one instance of getter
 *  for each corresponding C++ type.
 *
 *  @see ConverterMap
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 */

class Converter {
public:

  virtual ~Converter() {}

  /**
   *  @brief Return type_info of the corresponding C++ type.
   */
  virtual const std::type_info* typeinfo() const = 0;

  /**
   *  @brief Return value of pdsdata::TypeId::Type enum a type or -1.
   */
  virtual int pdsTypeId() const { return -1; }

  /**
   *  @brief Get the type version.
   *
   *  This method will disappear at some point, it is only necessary
   *  for current implementation based on strings.
   */
  virtual int getVersion() const { return -1; }

  /**
   *  @brief Convert C++ object to Python
   *
   *  @param[in] vdata  Void pointer to C++ data.
   *  @return New reference
   */
  virtual PyObject* convert(const boost::shared_ptr<void>& vdata) const = 0;

  /**
   *  @brief Returns Python type of the objects produced by this converter.
   *
   *  Some special converters can return PyBaseObject_Type (object).
   *  
   *  @return Borrowed reference
   */
  virtual PyTypeObject* pyTypeObject() const = 0;

};

}

#endif // PSDDL_PYTHON_CONVERTER_H
