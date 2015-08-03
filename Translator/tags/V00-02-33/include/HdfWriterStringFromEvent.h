#ifndef HDFWRITERSTRINGFROMEVENT_H
#define HDFWRITERSTRINGFROMEVENT_H

#include "Translator/HdfWriterFromEvent.h"
#include "Translator/HdfWriterString.h"

namespace Translator {

/**
 *  @ingroup Translator
 *
 *  @brief class to write datasets for std::string's in the event store into hdf5 groups
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class HdfWriterStringFromEvent : public HdfWriterFromEvent {
 public:  
  virtual void make_datasets(DataTypeLoc dataTypeLoc,
                             hdf5pp::Group & srcGroup, 
                             const PSEvt::EventKey & eventKey, 
                             PSEvt::Event & evt, 
                             PSEnv::Env & env,
                             bool shuffle,
                             int deflate,
                             boost::shared_ptr<Translator::ChunkPolicy> chunkPolicy) {
    boost::shared_ptr<std::string> ptr = 
      getFromEventStore<std::string>(eventKey, dataTypeLoc, evt, env);
    DataSetCreationProperties dataSetCreationProperties(chunkPolicy, shuffle, deflate);
    m_writer.setDatasetCreationProperties(dataSetCreationProperties);
    m_writer.make_dataset(srcGroup.id());
  }

  virtual void store(DataTypeLoc dataTypeLoc,
                     hdf5pp::Group & srcGroup, 
                     const PSEvt::EventKey & eventKey, 
                     PSEvt::Event & evt, 
                     PSEnv::Env & env) {
    boost::shared_ptr<std::string> ptr = 
      getFromEventStore<std::string>(eventKey, dataTypeLoc, evt, env);
    if (not ptr) {
      MsgLog("HdfWriterStringFromEvent",error,"store: evenKey: " << eventKey << " not found");
      throw std::runtime_error("HdfWriterStringFromEvent: event key not found");
    }
    m_writer.store(srcGroup, *ptr);
  }

  virtual void store_at(DataTypeLoc dataTypeLoc,
                        long index, hdf5pp::Group & srcGroup, 
                        const PSEvt::EventKey & eventKey, 
                        PSEvt::Event & evt, 
                        PSEnv::Env & env) {
    throw ErrSvc::Issue(ERR_LOC, "HdfWriterStringFromEvent::store_at() not implemented");
  }
  
  virtual void append(DataTypeLoc dataTypeLoc,
                      hdf5pp::Group & srcGroup, 
                      const PSEvt::EventKey & eventKey, 
                      PSEvt::Event & evt, 
                      PSEnv::Env & env) {
    boost::shared_ptr<std::string> ptr = 
      getFromEventStore<std::string>(eventKey, dataTypeLoc, evt, env);
    m_writer.append(srcGroup.id(), *ptr);
  }

  virtual void addBlank(hdf5pp::Group & group) {
    throw ErrSvc::Issue(ERR_LOC, "HdfWriterStringFromEvent::addBlank() not implemented");
  }

  virtual void closeDatasets(hdf5pp::Group &group) { m_writer.closeDataset(group.id()); }

 private:
  HdfWriterString m_writer;
}; // class HdfWriterStringFromEvent

} // namespace Translator

#endif
