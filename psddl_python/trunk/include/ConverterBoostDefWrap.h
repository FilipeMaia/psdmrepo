#ifndef PSDDL_PYTHON_CONVERTERBOOSTDEFWRAP_H
#define PSDDL_PYTHON_CONVERTERBOOSTDEFWRAP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConverterBoostDefWrap.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "Converter.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace psddl_python {

/// @addtogroup psddl_python

/**
 *  @ingroup psddl_python
 *
 *  @brief Default implementation of the converter interface for objects with wrappers.
 *
 *  Conversion is done via Boost.Python object class. Instance of a template type
 *  is first wrapped into a Wrapper class instance and that instance is passed to object
 *  constructor, so there should be a boost converter defined for Wrapper type.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename T, typename Wrapper>
class ConverterBoostDefWrap : public Converter {
public:

  // Default constructor
  ConverterBoostDefWrap(int pdsTypeId = -1, int version = -1) : m_pdsTypeId(pdsTypeId), m_version(version) {}

  // Destructor
  virtual ~ConverterBoostDefWrap() {}

  /**
   *  @brief Return type_info of the corresponding C++ type.
   */
  virtual const std::type_info* typeinfo() const { return &typeid(T); }

  /**
   *  @brief Return value of pdsdata::TypeId::Type enum a type or -1.
   */
  virtual int pdsTypeId() const { return m_pdsTypeId; }

  /**
   *  @brief Get the type version.
   *
   *  This method will disappear at some point, it is only necessary
   *  for current implementation based on strings.
   */
  virtual int getVersion() const { return m_version; }

  /**
   *  @brief Convert C++ object to Python
   *
   *  @param[in] vdata  Void pointer to C++ data.
   */
  virtual PyObject* convert(const boost::shared_ptr<void>& vdata) const {
    const boost::shared_ptr<T>& result = boost::static_pointer_cast<T>(vdata);
    if (result) {
      boost::python::object obj((Wrapper(result)));
      Py_INCREF(obj.ptr());
      return obj.ptr();
    }
    Py_RETURN_NONE;
  }

protected:

private:

  int m_pdsTypeId;
  int m_version;

};

} // namespace psddl_python

#endif // PSDDL_PYTHON_CONVERTERBOOSTDEFWRAP_H
