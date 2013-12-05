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

typedef enum { inEvent, inConfigStore } DataTypeLoc;

template <class T>
void checkType(const PSEvt::EventKey &eventKey, const char *logger) {
  const std::type_info & templateType = typeid(T);
  if (templateType != *eventKey.typeinfo()) {
    MsgLog(logger, error, "eventKey: " << eventKey << " type mismatch with template type: " 
           << PSEvt::TypeInfoUtils::typeInfoRealName(&templateType));
  }
}

template <class T>
boost::shared_ptr<T> getFromEventStore(const PSEvt::EventKey &eventKey,
                                       DataTypeLoc dataTypeLoc,
                                       PSEvt::Event & evt, PSEnv::Env & env) {
  checkType<T>(eventKey, "HdfWriter");
  boost::shared_ptr<T> ptr;
  if (dataTypeLoc == inEvent) ptr = evt.get(eventKey.src(), eventKey.key()); 
  else if (dataTypeLoc == inConfigStore) ptr = env.configStore().get(eventKey.src());
  return ptr;
}

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

  virtual ~HdfWriterFromEvent() {};
};

std::ostream & operator<<(std::ostream &o, DataTypeLoc &dataTypeLoc);
 
} // namespace

#endif
