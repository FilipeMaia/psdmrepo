/* The module defined below is for testing.
   It is a simple example that extracts cspad::DataV2 from 
   a hard coded source and writes to a hard coded h5 filename.
   It can be modified to test the psddl_hdf2psana dataset creation 
   and writing functions for different Psana types.
*/

#include "psana/Module.h"
#include "MsgLogger/MsgLogger.h"
#include "hdf5pp/File.h"
#include "hdf5pp/Group.h"
#include "psddl_hdf2psana/DefaultChunkPolicy.h"
#include "psddl_hdf2psana/cspad.ddl.h"

using namespace std;

const char * SRC = "DetInfo(CxiDs1.0:Cspad.0)";
const char * H5OUTPUT_FILE_NAME = "TestModulePsanaWrite.h5";

hsize_t chunkSizeTargetBytes = 16*1024*1024;
int chunkSizeTarget = 0;
hsize_t maxChunkSizeBytes = 100*1024*1024;
int minObjectsPerChunk = 50;
int maxObjectsPerChunk = 2048;
hsize_t minChunkCacheSize = 1024*1024;
hsize_t maxChunkCacheSize = 100*1024*1024;

class TestModulePsanaWriting : public Module {
public:
  psddl_hdf2psana::DefaultChunkPolicy m_chunkPolicy;
  
  TestModulePsanaWriting(std::string moduleName) : Module(moduleName), 
                                                   m_chunkPolicy(chunkSizeTargetBytes,
                                                                 chunkSizeTarget,
                                                                 maxChunkSizeBytes,
                                                                 minObjectsPerChunk,
                                                                 maxObjectsPerChunk,
                                                                 minChunkCacheSize,
                                                                 maxChunkCacheSize) {}

  hdf5pp::File m_h5file;
  hdf5pp::Group m_group;
  bool m_firstEvent;

  virtual void beginJob(Event& evt, Env& env) {
    hdf5pp::File::CreateMode mode = hdf5pp::File::Truncate;
    m_h5file = hdf5pp::File::create(H5OUTPUT_FILE_NAME, mode);
    m_group = m_h5file.createGroup("group");
    m_firstEvent = true;
  }

  virtual void event(Event& evt, Env& env) {
    PSEvt::Source source(SRC);
    boost::shared_ptr<Psana::CsPad::DataV2> cspad = evt.get(source);
    MsgLog(name(),info,"cspad ptr = " << cspad.get());
    const int latestTypeSchema = -1;
    if (m_firstEvent) {
      const int deflate = 1;
      const bool shuffle = true;
      psddl_hdf2psana::CsPad::make_datasets(*cspad, m_group, m_chunkPolicy, deflate, shuffle, latestTypeSchema);
      m_firstEvent = false;
    }
    const int indexForAppend = -1;
    psddl_hdf2psana::CsPad::store_at(cspad.get(), m_group, indexForAppend, latestTypeSchema);
  }

  virtual void endJob(Event& evt, Env& env) {
    m_group.close();
    m_h5file.close();
  }
};

PSANA_MODULE_FACTORY(TestModulePsanaWriting);


