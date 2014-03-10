////////////////////
// The module defined below is for testing.  It puts ndarrays and std::strings 
// into the event queue. The module name is TestModuleNDArrayString.
//
// Below are the key's and values for the data added.  a 1-up counter fo the event is used to 
// for the values.  Each of the nd arrays will set element 0 to the event number, and go 
// one up from there.  That is, for event 1, we'll have
//
//   type=std::string key="my_string1"              This is event number: 1
//   type=std::string key="my_string2"              This is a second string.  10 * event number is 10
//   EventKey type=ndarray<float,2>  "my_float2Da"  [[1.0,2.0],[3.0,4.0]]
//   EventKey type=ndarray<float,2>  "my_float2Db"  [[1.0,2.0],[3.0,4.0]]
//   EventKey type=ndarray<double,3> "my_double3D"  [[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]]
//   EventKey type=ndarray<int,1>    "my_int1D"     [1,2]
//   
//  
//  it takes the following config parameters to test error conditions
//  
//  vary_array_sizes = false       # this causes the array dimensions from one
//                                 # event to the next to differ, the Translator
//                                 # should see this and produce an error
//  use_fortran_stride = false      # this creates arrays with a fortran stride
//                                 # which is presently not supported in the Translator

#include <vector>
#include <sstream>

#include "psana/Module.h"

using namespace std;

namespace Translator {
  
class TestModuleNDArrayString : public Module {
public:
  TestModuleNDArrayString(std::string moduleName) : Module(moduleName) 
  {
    m_vary_array_sizes = config("vary_array_sizes", false);
    m_use_fortran_stride = config("use_fortran_stride", false);
  }

  virtual void beginJob(Event& evt, Env& env) {
    m_eventCounter = 0;
  }

  virtual void event(Event& evt, Env& env) {
    // put two strings in event queue
    ++m_eventCounter;
    ostringstream s1, s2;
    s1 << "This is event number: " << m_eventCounter;
    s2 << "This is a second string.  10 * event number is " << 10*m_eventCounter;
    boost::shared_ptr<std::string> string1Ptr = boost::make_shared<std::string>(s1.str());
    boost::shared_ptr<std::string> string2Ptr = boost::make_shared<std::string>(s2.str());
    evt.put(string1Ptr,"my_string1");
    evt.put(string2Ptr,"my_string2");

    // put 4 nd arrays in event queue
    const unsigned DIM = 2;
    const unsigned DIM1 = std::min(20UL, DIM + (m_vary_array_sizes ? m_eventCounter : 0));
    const unsigned DIM2 = DIM;
    const unsigned DIM3 = DIM;
    const unsigned twoDshape[2] = {DIM1,DIM2};
    const unsigned threeDshape[3] = {DIM1,DIM2,DIM3};
    const unsigned oneDshape[1] = {DIM1};
    
    const int numberOfNDarrays = 4;
    const char *keys[numberOfNDarrays] = {"my_float2Da", "my_float2Db", "my_double3D", "my_int1D"};
    const char types[numberOfNDarrays] = {'f', 'f', 'd', 'i'};
    const int NDims[numberOfNDarrays] = {2,2,3,1};
    const unsigned *shapes[numberOfNDarrays] = {twoDshape, twoDshape, threeDshape, oneDshape}; 
    
    for (int i = 0; i < numberOfNDarrays; ++i) {
      const char *key = keys[i];
      const int NDim = NDims[i];
      const unsigned *shape = shapes[i];
      switch (types[i]) {
        case 'f':
          if (NDim==1) createAndFillNDarrayThenPutInEvent<float,1>(shape, m_eventCounter, evt, key);
          if (NDim==2) createAndFillNDarrayThenPutInEvent<float,2>(shape, m_eventCounter, evt, key);
          if (NDim==3) createAndFillNDarrayThenPutInEvent<float,3>(shape, m_eventCounter, evt, key);
          break;
        case 'd':
          if (NDim==1) createAndFillNDarrayThenPutInEvent<double,1>(shape, m_eventCounter, evt, key);
          if (NDim==2) createAndFillNDarrayThenPutInEvent<double,2>(shape, m_eventCounter, evt, key);
          if (NDim==3) createAndFillNDarrayThenPutInEvent<double,3>(shape, m_eventCounter, evt, key);
          break;
        case 'i':
          if (NDim==1) createAndFillNDarrayThenPutInEvent<int,1>(shape, m_eventCounter, evt, key);
          if (NDim==2) createAndFillNDarrayThenPutInEvent<int,2>(shape, m_eventCounter, evt, key);
          if (NDim==3) createAndFillNDarrayThenPutInEvent<int,3>(shape, m_eventCounter, evt, key);
          break;
        }
    } // end adding all the arrays
  }
protected:
  template < class T, unsigned NDim>
  void createAndFillNDarrayThenPutInEvent(const unsigned *shape, size_t fillValue, Event &evt, const char *key) {
    enum ndns::Order order = ndns::C;
    if (m_use_fortran_stride) order = ndns::Fortran;
    boost::shared_ptr< ndarray<T,NDim> > arrayPtr = boost::make_shared< ndarray<T,NDim> >(shape, order);
    T * element = arrayPtr->data();
    for (unsigned i = 0; i < arrayPtr->size(); ++i) *element++ = T(fillValue + i);
    evt.put(arrayPtr, key);
  }
private:
  size_t m_eventCounter;
  bool m_use_fortran_stride, m_vary_array_sizes;
};

PSANA_MODULE_FACTORY(TestModuleNDArrayString);

}
