#ifndef PSDDL_PYTHON_GETTERMAP_H
#define PSDDL_PYTHON_GETTERMAP_H

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
#include <psddl_python/Getter.h>

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
 *  @brief Collection of the gtter objects indexed by C++ types.
 *
 *  @see Getter
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 */

class GetterMap {
public:

  typedef std::vector<std::string> NameList;

  /**
   *  @brief Returns singleton instance.
   */
  static GetterMap& instance();

  /**
   *  @brief Add one more getter instance to the map.
   */
  void addGetter(const boost::shared_ptr<Getter>& getter);

  /**
   *  @brief Find a getter for corresponding C++ type.
   */
  boost::shared_ptr<Getter> getGetter(const std::type_info& type) const;

  /**
   *  @brief Find a getter for corresponding C++ type name.
   */
  boost::shared_ptr<Getter> getGetter(const std::string& typeName) const;

  /**
   *  @brief Return matching template string for given type name
   */
  const NameList& getTemplate(const std::string& typeName) const;

protected:

  GetterMap() {}

private:

  void printTables();

  typedef std::tr1::reference_wrapper<const std::type_info> type_info_ref;
  struct TypeInfoCmp {
    bool operator()(const type_info_ref& lhs, const type_info_ref& rhs) const {
      return lhs.get().before(rhs.get());
    }
  };

  typedef std::map<std::string, boost::shared_ptr<Getter> > GetterNameMap;
  typedef std::map<type_info_ref, boost::shared_ptr<Getter>, TypeInfoCmp> GetterTypeMap;
  typedef std::map<std::string, NameList > TemplateMap;

  GetterNameMap m_getterNameMap;  // map C++ type name (version NOT removed) to getter
  GetterTypeMap m_getterTypeMap;  // map C++ type to getter
  TemplateMap m_templateMap; // map C++ type name (version IS removed) to list of names with versions
  NameList m_emptyNameList;

};

}

#endif // PSDDL_PYTHON_GETTERMAP_H
