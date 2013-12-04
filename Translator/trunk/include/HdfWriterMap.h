#ifndef TRANSLATOR_HDFWRITERMAP_H
#define TRANSLATOR_HDFWRITERMAP_H

#include <map>

#include "boost/shared_ptr.hpp"

#include "pdsdata/xtc/Src.hh"

#include "PSEvt/TypeInfoUtils.h"

#include "Translator/HdfWriterFromEvent.h"


namespace Translator {

typedef std::map<const std::type_info *, boost::shared_ptr<HdfWriterFromEvent> , PSEvt::TypeInfoUtils::lessTypeInfoPtr >  HdfWriterMap;

void initializeHdfWriterMap( HdfWriterMap & mapping);

boost::shared_ptr<HdfWriterFromEvent> 
getHdfWriter(HdfWriterMap &, const std::type_info *);

} // namespace

#endif
