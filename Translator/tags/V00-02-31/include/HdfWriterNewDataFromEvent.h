#ifndef TRANSLATOR_HDFWRITERNEWDATAFROMEVENT_H
#define TRANSLATOR_HDFWRITERNEWDATAFROMEVENT_H

#include <string>

#include "Translator/HdfWriterFromEvent.h"
#include "Translator/HdfWriterNew.h"
#include "Translator/HdfWriterGeneric.h"


namespace Translator {

/**
 *  @ingroup Translator
 *
 *  @brief class for writing user defined types
 *
 *  @note This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @author David Schneider
 */
class HdfWriterNewDataFromEvent : public HdfWriterFromEvent {
 public:
  HdfWriterNewDataFromEvent(const HdfWriterNew & newWriter, 
                            const std::string & key);

  HdfWriterNewDataFromEvent(boost::shared_ptr<HdfWriterNew> newWriter, 
                            const std::string & key);

  virtual void make_datasets(DataTypeLoc dataTypeLoc,
                             hdf5pp::Group & srcGroup, 
                             const PSEvt::EventKey & eventKey, 
                             PSEvt::Event & evt, 
                             PSEnv::Env & env,
                             bool shuffle,
                             int deflate,
                             boost::shared_ptr<Translator::ChunkPolicy> chunkPolicy);

  virtual void store(DataTypeLoc dataTypeLoc,
                     hdf5pp::Group & srcGroup, 
                     const PSEvt::EventKey & eventKey, 
                     PSEvt::Event & evt, 
                     PSEnv::Env & env);

  virtual void store_at(DataTypeLoc dataTypeLoc,
                        long index, hdf5pp::Group & srcGroup, 
                        const PSEvt::EventKey & eventKey, 
                        PSEvt::Event & evt, 
                        PSEnv::Env & env);
  
  virtual void append(DataTypeLoc dataTypeLoc,
                      hdf5pp::Group & srcGroup, 
                      const PSEvt::EventKey & eventKey, 
                      PSEvt::Event & evt, 
                      PSEnv::Env & env);

  virtual void addBlank(hdf5pp::Group & group);

  virtual void closeDatasets(hdf5pp::Group &group);

  virtual ~HdfWriterNewDataFromEvent();

  const std::string & key() { return m_key; }
 protected:
  bool checkTypeMatch(const PSEvt::EventKey & eventKey, std::string msg="");
  class NotImplementedException : public ErrSvc::Issue {
  public:
  NotImplementedException(const ErrSvc::Context &ctx, const std::string &what) : ErrSvc::Issue(ctx,what) {}
  }; // class NotImplementedException

 private:
  const std::type_info *m_typeInfoPtr; 
  std::string m_datasetName;
  HdfWriterNew::CreateHDF5Type m_createType;
  HdfWriterNew::FillHdf5WriteBuffer m_fillWriteBuffer;
  HdfWriterNew::CloseHDF5Type m_closeType;
  Translator::HdfWriterGeneric m_writer;
  std::string m_key;
}; // class HdfWriterNewDataFromEvent

} // namespace

#endif
