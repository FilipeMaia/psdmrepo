#ifndef TRANSLATOR_NDARRAYUTIL_H
#define TRANSLATOR_NDARRAYUTIL_H

#include <string>
#include <typeinfo>
#include "boost/shared_ptr.hpp"
#include "psddl_hdf2psana/NDArrayConverter.h"

namespace Translator {

boost::shared_ptr<const psddl_hdf2psana::NDArrayParameters>  
  ndarrayParameters(const std::type_info *ndarrayTypeInfoPtr, 
                    enum psddl_hdf2psana::NDArrayParameters::VlenDim vlenDim);

std::string ndarrayGroupName(const std::type_info *ndarrayTypeInfoPtr, 
                             enum psddl_hdf2psana::NDArrayParameters::VlenDim vlenDim);

} // namespace Translator

#endif
