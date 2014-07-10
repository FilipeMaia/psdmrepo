#ifndef TRANSLATOR_HDF5UTIL_H
#define TRANSLATOR_HDF5UTIL_H

#include <string>

#include "hdf5/hdf5.h"

namespace Translator {
namespace hdf5util {

void addAttribute_uint64(hid_t hid, const char * name, uint64_t val); 

std::string objectName(hid_t grp);

std::string type2str(hid_t typeId);

} // namespace hdf5util
} // namespace Translator

#endif

