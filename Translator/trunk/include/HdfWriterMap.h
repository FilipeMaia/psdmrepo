#ifndef TRANSLATOR_HDFWRITERMAP_H
#define TRANSLATOR_HDFWRITERMAP_H

#include <map>

#include "boost/shared_ptr.hpp"

#include "pdsdata/xtc/Src.hh"

#include "PSEvt/TypeInfoUtils.h"

#include "Translator/HdfWriterFromEvent.h"


namespace Translator {

typedef std::map<const std::type_info *, boost::shared_ptr<HdfWriterFromEvent> , PSEvt::TypeInfoUtils::lessTypeInfoPtr >  HdfWriterMap;

/// initializes the HdfWriterMap argument with writers for all the known Psana 
/// types, as well as a number of ndarrays, and std::string
void initializeHdfWriterMap( HdfWriterMap & mapping);

/// returns a pointer to the writer, or null if not available, for the given C++ type
boost::shared_ptr<HdfWriterFromEvent> 
getHdfWriter(HdfWriterMap &, const std::type_info *);

} // namespace

#endif
