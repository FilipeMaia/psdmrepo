#ifndef TRANSLATOR_H5GROUPNAMES_H
#define TRANSLATOR_H5GROUPNAMES_H

#include <string>
#include <typeinfo>
#include "pdsdata/xtc/Src.hh"
#include "Translator/TypeAliases.h"

namespace Translator {

class H5GroupNames {
 public:
  H5GroupNames(bool short_bld_name, 
               const TypeAliases::TypeInfoSet & ndarrays);
  std::string nameForType(const std::type_info *typeInfoPtr);
  std::string nameForSrc(const Pds::Src &src);
  bool isNDArray(const std::type_info *typeInfoPtr) { return m_ndarrays.find(typeInfoPtr) != m_ndarrays.end(); }
 private:
  bool m_short_bld_name;
  const TypeAliases::TypeInfoSet m_ndarrays;
}; // class H5GroupNames

} // namespace

#endif
