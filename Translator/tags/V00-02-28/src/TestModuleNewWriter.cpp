#include "psana/Module.h"
#include "MsgLogger/MsgLogger.h"
#include "Translator/HdfWriterNew.h"

  
////////////////////
// below is an example and for testing

namespace Translator {
  
struct MyData {
  int32_t eventCounter;
  float energy;
};

hid_t createMyDataHdf5Type(const void *) {
  static bool firstCall = true;
  static hid_t h5type = -1;
  if (not firstCall) return h5type;

  firstCall = false;
  h5type = H5Tcreate(H5T_COMPOUND, sizeof(MyData));
  
  herr_t status1 = H5Tinsert(h5type, "eventCounter", 
                             offsetof(MyData,eventCounter), 
                             H5T_NATIVE_UINT32);
  herr_t status2 = H5Tinsert(h5type, "energy", 
                             offsetof(MyData,energy), 
                             H5T_NATIVE_FLOAT);
  if ((h5type < 0) or (status1 < 0) or (status2<0)) {
    MsgLog("mydata",fatal,"unable to create MyData compound type");
  }
  MsgLog("mydata",trace,"Created hdf5 type for MyData  " << h5type);  
  return h5type;
}

const void * fillMyDataWriteBuffer(const void *data) {
  return data;
}

class TestNewHdfWriter : public Module {
public:

  TestNewHdfWriter(std::string moduleName) : Module(moduleName) {}

  virtual void beginJob(Event& evt, Env& env) {
    m_eventCounter = 0;
    MsgLog(name(),trace,"beginJob(), calling addHdfWriter");
    boost::shared_ptr<Translator::HdfWriterNew> newWriter = 
      boost::make_shared<Translator::HdfWriterNew>(&typeid(MyData), 
                                                   "data", 
                                                   createMyDataHdf5Type, 
                                                   fillMyDataWriteBuffer);
    evt.put(newWriter,name());
  }

  virtual void event(Event& evt, Env& env) {
    boost::shared_ptr<MyData> myData = boost::make_shared<MyData>();
    ++m_eventCounter;
    myData->eventCounter = m_eventCounter;
    myData->energy = 23.239;
    evt.put(myData,name());
  }
  private:
  size_t m_eventCounter;
};

PSANA_MODULE_FACTORY(TestNewHdfWriter);

} // namespace Translator
