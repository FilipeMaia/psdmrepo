#ifndef PSDDL_PYTHON_GETTER_H
#define PSDDL_PYTHON_GETTER_H

//--------------------------------------------------------------------------
// File and Version Information:
//      $Id: PyDataType.h 5266 2013-01-31 20:14:36Z salnikov@SLAC.STANFORD.EDU $
//
// Description:
//      Class Getter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <typeinfo>
#include <boost/shared_ptr.hpp>
#include <boost/python.hpp>

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
 *  @brief Class defining interface for "getter" object types.
 *
 *  Instances of getter types know how to extract data from proxy dictionary
 *  and convert it into Python objects. There will be one instance of getter
 *  for each corresponding C++ type.
 *
 *  @see GetterMap
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 */

class Getter {
public:

  virtual ~Getter() {}

  /**
   *  @brief Return type_info of the corresponding C++ type.
   */
  virtual const std::type_info& typeinfo() const = 0;

  /**
   *  @brief Return name of the corresponding C++ type.
   *
   *  There should be no assumptions about the names of the classes
   *  except for the uniqueness. This method is likely to disappear
   *  in the future when we switch to type-based system
   */
  virtual const char* getTypeName() const = 0;

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
   */
  virtual boost::python::object convert(const boost::shared_ptr<void>& vdata) const = 0;

};

}

#endif // PSDDL_PYTHON_GETTER_H
