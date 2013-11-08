#ifndef TRANSLATOR_HDFWRITERBASE_H
#define TRANSLATOR_HDFWRITERBASE_H

#include "boost/shared_ptr.hpp"
#include "hdf5pp/Group.h"
#include "PSEvt/EventKey.h"
#include "PSEvt/Event.h"
#include "PSEnv/Env.h"
#include "psddl_hdf2psana/ChunkPolicy.h"

namespace Translator {

typedef enum { inEvent, inConfigStore } DataTypeLoc;

class HdfWriterBase {
 public:
  virtual void make_datasets(DataTypeLoc dataTypeLoc,
                             hdf5pp::Group & srcGroup, 
                             const PSEvt::EventKey & eventKey, 
                             PSEvt::Event & evt, 
                             PSEnv::Env & env,
                             bool shuffle,
                             int deflate,
                             boost::shared_ptr<psddl_hdf2psana::ChunkPolicy> chunkPolicy) = 0;

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

  virtual ~HdfWriterBase() {};
};

std::ostream & operator<<(std::ostream &o, DataTypeLoc &dataTypeLoc);
 
} // namespace

#endif
