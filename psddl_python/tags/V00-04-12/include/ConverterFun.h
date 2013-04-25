#ifndef PSDDL_PYTHON_CONVERTERFUN_H
#define PSDDL_PYTHON_CONVERTERFUN_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConverterFun.
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
 *  @brief Default implementation of the converter interface for value-type objects.
 *
 *  Conversion is done via a user-provided conversion function which should be
 *  passed to converter constructor.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename T, typename Functor>
class ConverterFun : public Converter {
public:

  // Default constructor
  ConverterFun(Functor func, PyTypeObject* pyTypeObject, int pdsTypeId = -1, int version = -1)
    : m_func(func), m_pyTypeObject(pyTypeObject), m_pdsTypeId(pdsTypeId), m_version(version) 
  {
    Py_INCREF(m_pyTypeObject);
  }

  // Destructor
  virtual ~ConverterFun() 
  {
    Py_CLEAR(m_pyTypeObject);
  }

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
   *  @return New reference
   */
  virtual PyObject* convert(const boost::shared_ptr<void>& vdata) const {
    const boost::shared_ptr<T>& result = boost::static_pointer_cast<T>(vdata);
    if (result) return m_func(result);
    Py_RETURN_NONE;
  }

  /**
   *  @brief Returns Python type of the objects produced by this converter.
   *
   *  Some special converters can return PyBaseObject_Type (object).
   *  
   *  @return Borrowed reference
   */
  virtual PyTypeObject* pyTypeObject() const
  {
    return m_pyTypeObject;
  }

protected:

private:

  Functor m_func;
  PyTypeObject* m_pyTypeObject;
  int m_pdsTypeId;
  int m_version;

};

/**
 *  helper function to make instances of the above class
 */
template <typename T, typename Functor>
boost::shared_ptr<ConverterFun<T, Functor> >
make_converter_fun(Functor func, PyTypeObject* pyTypeObject, int pdsTypeId = -1, int version = -1)
{
  return boost::make_shared<ConverterFun<T, Functor> >(func, pyTypeObject, pdsTypeId, version);
}

} // namespace psddl_python

#endif // PSDDL_PYTHON_CONVERTERFUN_H
