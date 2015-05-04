#ifndef TRANSLATOR_TYPEALIAS_MAP_H
#define TRANSLATOR_TYPEALIAS_MAP_H

#include <string>
#include <map>
#include <set>
#include <typeinfo>
#include "PSEvt/TypeInfoUtils.h"

namespace Translator {

  /**
   * @ingroup TypeAlias
   *
   * @brief class providing aliases to refer to sets of Psana types.
   * 
   * For example, 'CsPad' is an alias for all of the Config and Data CsPad types.  
   * The intention it to provide aliases that make it easy to filter type
   * without using the C++ type names.  The groupings are listed below.  They
   * are generated from the DDL, so they may not be perfect.  For instance
   * if one needs to distinguish between CsPad::DataV1 and CsPad::DataV2, 
   * you cannot use these aliases.
   * 
   * The most current alias list is generated in the file data/default_psana.cfg of
   * the Translator package directory.
   */

class TypeAliases {
public:
  TypeAliases();
  typedef std::set<const std::type_info *, PSEvt::TypeInfoUtils::lessTypeInfoPtr> TypeInfoSet;
  typedef std::map<std::string, TypeInfoSet > Alias2TypesMap;
  typedef std::map<const std::type_info *, std::string,  PSEvt::TypeInfoUtils::lessTypeInfoPtr > Type2AliasMap;

  const std::set<std::string> & aliases() { return m_aliasKeys; }
  const Alias2TypesMap & alias2TypesMap(){ return m_alias2TypesMap; }
  const Type2AliasMap & type2AliasMap(){ return m_type2AliasMap; }
private:
  std::set<std::string> m_aliasKeys;
  Alias2TypesMap m_alias2TypesMap;
  Type2AliasMap m_type2AliasMap;
};

} // namespace

#endif
