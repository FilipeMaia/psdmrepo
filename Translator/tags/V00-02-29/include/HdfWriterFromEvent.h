#ifndef TRANSLATOR_HDFWRITERFROMEVENT_H
#define TRANSLATOR_HDFWRITERFROMEVENT_H

#include "boost/shared_ptr.hpp"
#include "hdf5pp/Group.h"
#include "PSEvt/EventKey.h"
#include "PSEvt/Event.h"
#include "PSEvt/TypeInfoUtils.h"
#include "PSEnv/Env.h"
#include "Translator/ChunkPolicy.h"
#include "MsgLogger/MsgLogger.h"

namespace Translator {

typedef enum { inEvent, inConfigStore, inCalibStore } DataTypeLoc;

/// helper function for HdfWriterFromEvent 
template <class T>
void checkType(const PSEvt::EventKey &eventKey, const char *logger) {
  const std::type_info & templateType = typeid(T);
  if (templateType != *eventKey.typeinfo()) {
    MsgLog(logger, error, "eventKey: " << eventKey << " type mismatch with template type: " 
           << PSEvt::TypeInfoUtils::typeInfoRealName(&templateType));
  }
}

/// helper function for HdfWriterFromEvent 
template <class T>
boost::shared_ptr<T> getFromEventStore(const PSEvt::EventKey &eventKey,
                                       DataTypeLoc dataTypeLoc,
                                       PSEvt::Event & evt, PSEnv::Env & env) {
  checkType<T>(eventKey, "HdfWriter");
  boost::shared_ptr<T> ptr;
  switch (dataTypeLoc) {
  case inEvent: 
    ptr = evt.get(eventKey.src(), eventKey.key()); 
    break;
  case inConfigStore:
    ptr = env.configStore().get(eventKey.src(), eventKey.key());
    break;
  case inCalibStore:
    ptr = env.calibStore().get(eventKey.src(), eventKey.key());
    break;
  }
  return ptr;
}

/**
 *  @ingroup Translator
 *
 *  @brief interface for classes that write an Object in the Event, or config store.
 *
 *  Base class for hdf writer classes that retrieve an object from the event or 
 *  config store.  A derived class may implement writing details either using the 
 *  hdf5 writing functions in psddl_hdf2psana (such as store_at, store) or a 
 *  Translator/HdfWriterGeneric class.
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class HdfWriterFromEvent {
 public:
  virtual void make_datasets(DataTypeLoc dataTypeLoc,
                             hdf5pp::Group & srcGroup, 
                             const PSEvt::EventKey & eventKey, 
                             PSEvt::Event & evt, 
                             PSEnv::Env & env,
                             bool shuffle,
                             int deflate,
                             boost::shared_ptr<Translator::ChunkPolicy> chunkPolicy) = 0;

  virtual void store(DataTypeLoc dataTypeLoc,
                     hdf5pp::Group & srcGroup, 
                     const PSEvt::EventKey & eventKey, 
                     PSEvt::Event & evt, 
                     PSEnv::Env & env) = 0;

  virtual void store_at(DataTypeLoc dataTypeLoc,
                        long index, hdf5pp::Group & srcGroup, 
                        const PSEvt::EventKey & eventKey, 
                        PSEvt::Event & evt, 
                        PSEnv::Env & env) = 0;
  
  virtual void append(DataTypeLoc dataTypeLoc,
                      hdf5pp::Group & srcGroup, 
                      const PSEvt::EventKey & eventKey, 
                      PSEvt::Event & evt, 
                      PSEnv::Env & env) = 0;

  virtual void addBlank(hdf5pp::Group & group) = 0;

  // psddl_hdf2psana based writers should not implement close dataset operations,
  // datasets are closed by the hdf5pp::Group object.  However HdfWriterGeneric based 
  // writers need to implement these functions.
  virtual void closeDatasets(hdf5pp::Group &group) {}

  virtual ~HdfWriterFromEvent() {};
};

std::ostream & operator<<(std::ostream &o, DataTypeLoc &dataTypeLoc);
 
} // namespace

#endif
