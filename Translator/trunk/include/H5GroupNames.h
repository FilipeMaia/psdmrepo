#ifndef TRANSLATOR_H5GROUPNAMES_H
#define TRANSLATOR_H5GROUPNAMES_H

#include <string>
#include <typeinfo>
#include "pdsdata/xtc/Src.hh"
#include "Translator/TypeAliases.h"

namespace Translator {

/**
 *  @ingroup Translator
 *
 *  @brief transforms C++ Psana types and src locations into hdf5 group names.
 *
 *  Also returns true if a C++ type is a NDArray, requires the set of 
 *  NDArray types recognized in the system to be passed in for this.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class H5GroupNames {
 public:
  H5GroupNames(bool short_bld_name, 
               const TypeAliases::TypeInfoSet & ndarrays);
  std::string nameForType(const std::type_info *typeInfoPtr);
  std::string nameForSrc(const Pds::Src &src);
  bool isNDArray(const std::type_info *typeInfoPtr) { 
    return m_ndarrays.find(typeInfoPtr) != m_ndarrays.end(); 
  }
 private:
  bool m_short_bld_name;
  const TypeAliases::TypeInfoSet m_ndarrays;
}; // class H5GroupNames

} // namespace

#endif
