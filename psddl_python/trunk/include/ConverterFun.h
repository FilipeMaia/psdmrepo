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
  ConverterFun(Functor func, PyTypeObject* pyTypeObject, int pdsTypeId = -1)
    : m_func(func), m_pyTypeObject(pyTypeObject), m_pdsTypeId(pdsTypeId)
  {
    Py_INCREF(m_pyTypeObject);
  }

  // Destructor
  virtual ~ConverterFun() 
  {
    Py_CLEAR(m_pyTypeObject);
  }

  /**
   *  @brief Return type_infos of source C++ types.
   *
   *  If converter supports conversion from C++ to Python
   *  this method shall return non-empty vector.
   */
  virtual std::vector<const std::type_info*> from_cpp_types() const
  {
    return std::vector<const std::type_info*>(1, &typeid(T));
  }

  /**
   *  @brief Returns source Python types.
   *
   *  If converter supports conversion from Python to C++
   *  this method shall return non-empty vector.
   *
   *  @return Borrowed references
   */
  virtual std::vector<PyTypeObject*> from_py_types() const
  {
    return std::vector<PyTypeObject*>(1, m_pyTypeObject);
  }

  /**
   *  @brief Returns destination Python types.
   *
   *  If converter supports conversion from C++ to Python
   *  this method shall return non-empty vector.
   *
   *  @return Borrowed references
   */
  virtual std::vector<PyTypeObject*> to_py_types() const { return from_py_types(); }

  /**
   *  @brief Return list of pdsdata::TypeId::Type enum values.
   */
  virtual std::vector<int> from_pds_types() const {
    std::vector<int> res;
    if (m_pdsTypeId >= 0) res.push_back(m_pdsTypeId);
    return res;
  }

  /**
   *  @brief Convert C++ object to Python
   *
   *  @return New reference
   */
  virtual PyObject* convert(PSEvt::ProxyDictI& proxyDict, const PSEvt::Source& source, const std::string& key) const {
    const boost::shared_ptr<void>& vdata = proxyDict.get(&typeid(T), source, key, 0);
    const boost::shared_ptr<T>& cppobj = boost::static_pointer_cast<T>(vdata);
    if (cppobj) return m_func(cppobj);
    return 0;
  }

protected:

private:

  Functor m_func;
  PyTypeObject* m_pyTypeObject;
  int m_pdsTypeId;

};

/**
 *  helper function to make instances of the above class
 */
template <typename T, typename Functor>
boost::shared_ptr<ConverterFun<T, Functor> >
make_converter_fun(Functor func, PyTypeObject* pyTypeObject, int pdsTypeId = -1)
{
  return boost::make_shared<ConverterFun<T, Functor> >(func, pyTypeObject, pdsTypeId);
}

} // namespace psddl_python

#endif // PSDDL_PYTHON_CONVERTERFUN_H
