#ifndef TRANSLATOR_HDFWRITERMAP_H
#define TRANSLATOR_HDFWRITERMAP_H

#include <map>

#include "boost/shared_ptr.hpp"

#include "pdsdata/xtc/Src.hh"

#include "PSEvt/TypeInfoUtils.h"

#include "Translator/HdfWriterBase.h"


namespace Translator {

  typedef std::map<const std::type_info *, boost::shared_ptr<HdfWriterBase> , PSEvt::TypeInfoUtils::lessTypeInfoPtr >  HdfWriterMap;

void initializeHdfWriterMap( HdfWriterMap & );

boost::shared_ptr<HdfWriterBase> 
getHdfWriter(HdfWriterMap &, const std::type_info *);

} // namespace

#endif
