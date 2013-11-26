#include <iostream>

#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
#include "ndarray/ndarray.h"
#include "PSEvt/Event.h"
#include "PSEvt/ProxyDict.h"
#include "hdf5pp/File.h"
#include "hdf5pp/Group.h"

using namespace std;

const string arrayKeyString = "myarray";

struct A {
  int i;
  float x;
};

void foo() {
  ndarray<A,1> data = make_ndarray<A>(3);
}

void loadEvent(boost::shared_ptr<PSEvt::Event> evt) {
  const unsigned DIM1 = 10;
  const unsigned DIM2 = 12;
  unsigned int shape[2] = {DIM1,DIM2};
  boost::shared_ptr< ndarray<int,2> > myarray = boost::make_shared<ndarray<int,2> >(shape);
  ndarray<A,1> data = make_ndarray<A>(10);
  data[0].i=3;
  data[0].x=3234.3;
  data[19].i=3;
  data[19].x=34.0;
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
  boost::shared_ptr<PSEvt::ProxyDictI> proxyDict = boost::make_shared<PSEvt::ProxyDict>();
  boost::shared_ptr<PSEvt::Event> evt = boost::make_shared<PSEvt::Event>(proxyDict);
  loadEvent(evt);
  boost::shared_ptr< ndarray<int,2> > myarray = evt->get(arrayKeyString);
  cout << "myarray.get(): " << myarray.get() << endl; 
  printArray(cout,*myarray);
  hdf5pp::File::CreateMode mode = hdf5pp::File::Truncate;
  hdf5pp::File h5file = hdf5pp::File::create(m_h5fileName, mode);
  hdf5pp::Group group = h5file.createGroup("group");
  
    
  

  
  return 0;
}
