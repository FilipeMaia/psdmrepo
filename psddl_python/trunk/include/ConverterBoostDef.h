#ifndef PSDDL_PYTHON_CONVERTERBOOSTDEF_H
#define PSDDL_PYTHON_CONVERTERBOOSTDEF_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConverterBoostDef.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "psddl_python/Converter.h"

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
 *  Conversion is done via Boost.Python object class. Instance of a template type
 *  is passed to obejct constructor directly, so there should be a boost converter
 *  defined for this type.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

template <typename T>
class ConverterBoostDef : public Converter {
public:

  // Default constructor
  ConverterBoostDef(int pdsTypeId = -1) : m_pdsTypeId(pdsTypeId) {}

  // Destructor
  virtual ~ConverterBoostDef() {}

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
    // find registration info for type T
    boost::python::converter::registration const* reg = boost::python::converter::registry::query(boost::python::type_id<T>());
    if (reg) {
      try {
        return std::vector<PyTypeObject*>(1, reg->get_class_object());
      } catch (const boost::python::error_already_set& ex) {
        // exception was likely generated, clear it
        PyErr_Clear();
      }
    }
    return std::vector<PyTypeObject*>();
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
    if (cppobj) {
      boost::python::object obj(*cppobj);
      Py_INCREF(obj.ptr());
      return obj.ptr();
    }
    // 0 does not mean python error
    return 0;
  }

protected:

private:

  int m_pdsTypeId;

};

} // namespace psddl_python

#endif // PSDDL_PYTHON_CONVERTERBOOSTDEF_H
