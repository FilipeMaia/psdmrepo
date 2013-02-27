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
#include <tr1/functional>
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

  typedef std::vector<std::string> NameList;
  typedef std::vector<const std::type_info*> TypeInfoList;

  /**
   *  @brief Returns singleton instance.
   */
  static ConverterMap& instance();

  /**
   *  @brief Add one more converter instance to the map.
   */
  void addConverter(const boost::shared_ptr<Converter>& cvt);

  /**
   *  @brief Find a converter for corresponding C++ type.
   */
  boost::shared_ptr<Converter> getConverter(const std::type_info* type) const;

  /**
   *  @brief Return list of type names matching Pds::TypeId::Type value 
   */
  const TypeInfoList& pdsTypeInfos(int pdsTypeId) const;

protected:

  ConverterMap() {}

private:

  struct TypeInfoCmp {
    bool operator()(const std::type_info* lhs, const std::type_info* rhs) const {
      return lhs->before(*rhs);
    }
  };

  typedef std::map<const std::type_info*, boost::shared_ptr<Converter>, TypeInfoCmp> ConverterTypeMap;
  typedef std::map<int, TypeInfoList> PdsTypeMap;

  ConverterTypeMap m_cvtTypeMap;  // map C++ type to converter
  PdsTypeMap m_pdsTypeMap;        // map TypeId number to a list of typeinfos
  TypeInfoList m_emptyTypeList;

};

}

#endif // PSDDL_PYTHON_CONVERTERMAP_H
