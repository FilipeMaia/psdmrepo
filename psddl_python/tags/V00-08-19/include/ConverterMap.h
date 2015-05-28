#ifndef PSDDL_PYTHON_CONVERTERMAP_H
#define PSDDL_PYTHON_CONVERTERMAP_H

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
#include <map>
#include <string>
#include <vector>
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "psddl_python/Converter.h"

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
 *  @brief Collection of the converter objects indexed by C++ types.
 *
 *  @see Converter
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 */

class ConverterMap {
public:

  typedef boost::shared_ptr<Converter> mapped_type;
  typedef std::vector<mapped_type> CvtList;


  /**
   *  @brief Returns singleton instance.
   */
  static ConverterMap& instance();

  /**
   *  @brief Add one more converter instance to the map.
   */
  void addConverter(const boost::shared_ptr<Converter>& cvt);

  /**
   *  @brief Find converters for corresponding source C++ type.
   */
  const CvtList& getFromCppConverters(const std::type_info* type) const;

  /**
   *  @brief Find converters for corresponding source Python type.
   */
  const CvtList& getFromPyConverters(const PyTypeObject* type) const;

  /**
   *  @brief Find converters for corresponding destination Python type.
   */
  const CvtList& getToPyConverters(const PyTypeObject* type) const;

  /**
   *  @brief Returns set of converters for given PDS type ID.
   */
  const CvtList& getFromPdsConverters(int pdsTypeId) const;

protected:

  ConverterMap() {}

private:

  struct TypeInfoCmp {
    bool operator()(const std::type_info* lhs, const std::type_info* rhs) const {
      return lhs->before(*rhs);
    }
  };

  typedef std::map<const std::type_info*, CvtList, TypeInfoCmp> CppTypeMap;
  typedef std::map<const PyTypeObject*, CvtList> PyTypeMap;
  typedef std::map<int, CvtList> PdsTypeIdMap;

  CppTypeMap m_from_cpp_types;        // map source C++ type to converter
  PyTypeMap m_from_py_types;           // map source Python type to converter
  PyTypeMap m_to_py_types;             // map destination Python type to converter
  PdsTypeIdMap m_from_pds_types;       // map TypeId number to a list of converters
  CvtList m_emptyCvtList;

};

}

#endif // PSDDL_PYTHON_CONVERTERMAP_H
