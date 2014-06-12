#include <iostream>

#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
#include "ndarray/ndarray.h"
#include "PSEvt/Event.h"
#include "PSEvt/ProxyDict.h"
#include "hdf5pp/File.h"
#include "hdf5pp/Group.h"
#include "Translator/HdfWriterNDArray.h"
#include "MsgLogger/MsgLogger.h"

using namespace std;

const string arrayKeyString = "myarray";

namespace {
  const char * logger = "write-ndarray-to-h5-example";
}

void loadEvent(boost::shared_ptr<PSEvt::Event> evt) {
  const unsigned DIM1 = 10;
  const unsigned DIM2 = 12;
  unsigned int shape[2] = {DIM1,DIM2};
  boost::shared_ptr< ndarray<int,2> > myarray = boost::make_shared<ndarray<int,2> >(shape);
  int k = 0;
  for (unsigned i = 0; i < DIM1; ++i) {
    for (unsigned j = 0; j < DIM2; ++j) {
      (*myarray)[i][j]=k++;
    }
  }
  evt->put(myarray, arrayKeyString);
}

void printArray(ostream & o, const ndarray<int,2> &array) {
  const unsigned * shape = array.shape();
  const unsigned dim1 = shape[0];
  const unsigned dim2 = shape[1];
  for (unsigned i = 0; i < dim1; ++i) {
    for (unsigned j = 0; j < dim2; ++j) {
      cout << " " << array[i][j];
    }
    cout << endl;
  }
}

int main() {
  boost::shared_ptr<PSEvt::AliasMap> amap= boost::make_shared<PSEvt::AliasMap>();
  boost::shared_ptr<PSEvt::ProxyDictI> proxyDict = boost::make_shared<PSEvt::ProxyDict>(amap);
  boost::shared_ptr<PSEvt::Event> evt = boost::make_shared<PSEvt::Event>(proxyDict);
  boost::shared_ptr<PSEnv::IExpNameProvider> expNameProvider;
  boost::shared_ptr<PSEnv::Env> env = boost::make_shared<PSEnv::Env>("", expNameProvider, "", amap, 0);
  MsgLog(logger, info, "loading event with ndarray");
  loadEvent(evt);
  boost::shared_ptr< ndarray<int,2> > myarray = evt->get(arrayKeyString);
  WithMsgLog(logger, info, str) {
    str << "Extracted array from event store, pointer is:  " << myarray.get() << endl; 
    printArray(str,*myarray);
  }
  hdf5pp::File::CreateMode mode = hdf5pp::File::Truncate;
  const char * h5outputFile = "write-ndarray-to-h5-example.h5";
  hdf5pp::File h5file = hdf5pp::File::create(h5outputFile, mode);
  hdf5pp::Group group = h5file.createGroup("group");
  Translator::HdfWriterNDArray<int,2,false> writer;
  PSEvt::EventKey eventKey(&typeid(ndarray<int,2>),PSEvt::EventKey::anySource(),arrayKeyString);
  boost::shared_ptr<Translator::ChunkPolicy> chunkPolicy = boost::make_shared<Translator::ChunkPolicy>();
  writer.make_datasets(Translator::inEvent, group, eventKey, *evt, *env, false,-1,chunkPolicy);
  writer.append(Translator::inEvent, group, eventKey, *evt, *env);
  writer.append(Translator::inEvent, group, eventKey, *evt, *env);
  group.close();
  h5file.close();
  MsgLog(logger,info, "wrote h5file: " << h5outputFile);
  return 0;
}
