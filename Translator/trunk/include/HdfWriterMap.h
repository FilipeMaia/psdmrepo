#ifndef TRANSLATOR_HDFWRITERMAP_H
#define TRANSLATOR_HDFWRITERMAP_H

#include <map>
#include <vector>

#include "boost/shared_ptr.hpp"

#include "pdsdata/xtc/Src.hh"

#include "PSEvt/TypeInfoUtils.h"

#include "Translator/HdfWriterFromEvent.h"
#include "Translator/TypeClassEnum.h"

namespace Translator {

  /// maintains two maps - main one, and vlen.
  /// presently vlen map is just for ndarrays, these should be hdfwriters that
  /// allow for variable length on the slow dimension.
  class HdfWriterMap {    
  public:
    HdfWriterMap() {}
    void initialize();

    /// returns true if given type existed in one of the maps before being removed. 
    /// False if type was existed in neither map.
    /// Removes type from both main and vlen map.
    bool remove(const std::type_info * typeInfoPtr);

    /// replaces hdfwriter for given type from one of the maps.
    /// Defaults to replace on the main map. Optional argument specifies
    /// vlen map.
    /// return true if type was there before being replaced.
    bool replace(const std::type_info * typeInfoPtr,
                 boost::shared_ptr<HdfWriterFromEvent> hdfWriter, TypeClass typeClass) 
    {
      return replaceImpl(typeInfoPtr, hdfWriter, false, typeClass);
    };

    bool replaceVlen(const std::type_info * typeInfoPtr,
                     boost::shared_ptr<HdfWriterFromEvent> hdfWriter) {
      return replaceImpl(typeInfoPtr, hdfWriter, true, NdarrayType);
    }

    // return writer from main map by default or vlen map if specified.
    // also returns class of type - DAQ, ndarray, string, newWriter, if last
    // argument is not a NULL pointer
    boost::shared_ptr<HdfWriterFromEvent> find(const std::type_info * typeInfoPtr,  TypeClass *typeClass = NULL) {
      return findImpl(typeInfoPtr, false, typeClass);
    }

    boost::shared_ptr<HdfWriterFromEvent> findVlen(const std::type_info * typeInfoPtr) {
      return findImpl(typeInfoPtr, true, NULL);
    }
    
    // return list of types for given map - main (default) or vlen(optional arg).
    std::vector<const std::type_info *> types(bool vlen = false);

  private:    
    bool replaceImpl(const std::type_info * typeInfoPtr,
                     boost::shared_ptr<HdfWriterFromEvent>, bool vlen, TypeClass typeClass);

    boost::shared_ptr<HdfWriterFromEvent> findImpl(const std::type_info * typeInfoPtr,  
                                                   bool vlen, TypeClass *typeClass);

    typedef std::pair<boost::shared_ptr<HdfWriterFromEvent>, TypeClass> MapValue;
    typedef std::map<const std::type_info *,
      MapValue,
      PSEvt::TypeInfoUtils::lessTypeInfoPtr >  MapImpl;
    MapImpl m_mainMap;
    MapImpl m_vlenMap;
  };

} // namespace

#endif
