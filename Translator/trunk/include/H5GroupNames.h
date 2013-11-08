#ifndef TRANSLATOR_H5GROUPNAMES_H
#define TRANSLATOR_H5GROUPNAMES_H

#include <string>
#include <typeinfo>
#include "pdsdata/xtc/Src.hh"

namespace Translator {

std::string getH5GroupNameForType(const std::type_info *typeInfoPtr, bool short_bld_name=true);
std::string getH5GroupNameForSrc(const Pds::Src &src);

} // namespace

#endif
