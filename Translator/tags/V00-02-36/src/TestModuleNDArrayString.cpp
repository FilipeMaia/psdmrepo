////////////////////
// The file implements three psana modules for testing.
// These are:
//
// TestModuleNDArrayString
//
// TestModuleReadVlenNDArrayString
//
// TestModuleReadNoVlenNDArrayString 
//
// below we describe the modules:
//
// TestModuleNDArrayString
//
// This puts ndarrays and std::strings into the event queue. 
//
// Below are the key's and values for the non const ndarray's added, and the strings.  
// A 1-up counter fo the event is used to for the values.  Each of the 
// nd arrays will set element 0 to the event number (1-up), and go one up from there.  
// That is, for event 1, we'll have
//
//   type=std::string key="my_string1"              This is event number: 1
//   type=std::string key="my_string2"              This is a second string.  10 * event number is 10
//   EventKey type=ndarray<float,2>  "my_float2Da"  [[1.0,2.0],[3.0,4.0]]
//   EventKey type=ndarray<float,2>  "my_float2Db"  [[1.0,2.0],[3.0,4.0]]
//   EventKey type=ndarray<double,3> "my_double3D"  [[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]]
//   EventKey type=ndarray<int,1>    "my_int1D"     [1,2]
//   EventKey type=ndarray<unsigned,1> "my_uint1D"  [1,2]
//   
//  When we get to event 2, all the starting values will be 2. The strings will say 
//
//   type=std::string key="my_string1"              This is event number: 2
//   type=std::string key="my_string2"              This is a second string.  10 * event number is 20
//
//  Each dimensions of the arrays will be 2 unless vary_array_sizes is true.
//  In this case, the first dimension will be min(20, 2+eventNumber) where eventNumber is 1-up

//  The const data get 'c' prepended to the key:
//
//   EventKey type=ndarray<const float,2>  "cmy_float2Da"  [[1.0,2.0],[3.0,4.0]]
//   EventKey type=ndarray<const float,2>  "cmy_float2Db"  [[1.0,2.0],[3.0,4.0]]
//   EventKey type=ndarray<const double,3> "cmy_double3D"  [[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]]
//   EventKey type=ndarray<const int,1>    "cmy_int1D"     [1,2]
//   EventKey type=ndarray<const unsigned,1> "cmy_uint1D"  [1,2]
//   
//  
//  it takes the following config parameters to test error conditions
//  
//  vary_array_sizes = false       # this causes the array dimensions from one
//                                 # event to the next to differ, as described above
//                                 # should see this and produce an error
//  use_fortran_stride = false     # this creates arrays with a fortran stride
//                                 # which is presently not supported in the Translator
//  add_vlen_prefix = true         # will cause translate_vlen: to be added to the front of the keys
//  skip_event = 0                 # set to positive event number and module will call skip()
//                                 # use 1-up value for event
//
// The modules
// TestModuleReadVlenNDArrayString
//
// TestModuleReadNoVlenNDArrayString 
//  read the above data. They throw a fatal error if the data does not agree with 
//  what is expected.
//
#include <vector>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cmath>

#include "psana/Module.h"
#include "MsgLogger/MsgLogger.h"

#include "Translator/specialKeyStrings.h"

using namespace std;

namespace {
  template <class T>
  struct RemoveConst
  {
    typedef T type;
  };

  template <class T>
  struct RemoveConst<const T>
  {
    typedef T type;
  };
}

namespace Translator {
  
class TestModuleNDArrayString : public Module {
public:
  TestModuleNDArrayString(std::string moduleName) : Module(moduleName) 
  {
    m_vary_array_sizes = config("vary_array_sizes", false);
    m_use_fortran_stride = config("use_fortran_stride", false);
    m_vlen_prefix = config("vlen_prefix",false);
    m_ndarrayKeyPrefix = m_vlen_prefix ? string("translate_vlen:") : string();
    // option to skip one event (use 1-up event counter)
    m_skipEvent = config("skip_event",0);
  }

  virtual void beginJob(Event& evt, Env& env) {
    m_eventCounter = 0;
  }

  virtual void event(Event& evt, Env& env) {
    ++m_eventCounter;
    MsgLog(name(),debug,"Event number (1-up counter): " << m_eventCounter);
    if (m_skipEvent == int(m_eventCounter)) skip();
    // put two strings in event queue
    ostringstream s1, s2;
    s1 << "This is event number: " << m_eventCounter;
    s2 << "This is a second string.  10 * event number is " << 10*m_eventCounter;
    boost::shared_ptr<std::string> string1Ptr = boost::make_shared<std::string>(s1.str());
    boost::shared_ptr<std::string> string2Ptr = boost::make_shared<std::string>(s2.str());
    evt.put(string1Ptr, string("my_string1"));
    evt.put(string2Ptr, string("my_string2"));

    // put 5 nd arrays in event queue
    const unsigned DIM = 2;
    const unsigned DIM1 = std::min(20UL, DIM + (m_vary_array_sizes ? m_eventCounter : 0));
    const unsigned DIM2 = DIM;
    const unsigned DIM3 = DIM;
    const unsigned twoDshape[2] = {DIM1,DIM2};
    const unsigned threeDshape[3] = {DIM1,DIM2,DIM3};
    const unsigned oneDshape[1] = {DIM1};
    
    const int numberOfNDarrays = 5;
    const char *keys[numberOfNDarrays] = {"my_float2Da", "my_float2Db", "my_double3D", "my_int1D", "my_uint1D"};
    const char types[numberOfNDarrays] = {'f', 'f', 'd', 'i', 'u'};
    const int NDims[numberOfNDarrays] = {2,2,3,1,1};
    const unsigned *shapes[numberOfNDarrays] = {twoDshape, twoDshape, threeDshape, oneDshape, oneDshape}; 
    
    for (int i = 0; i < numberOfNDarrays; ++i) {
      const char *_key = keys[i];
      const int NDim = NDims[i];
      const unsigned *shape = shapes[i];
      for (int isConst = 0; isConst < 2; ++isConst) {
        string key = string(_key);
        if (isConst) key = "c" + key;
        key = m_ndarrayKeyPrefix + key;
        MsgLog(name(), debug, "  array " << i << " const=" << isConst << " key=" << key);
        switch (types[i]) {
        case 'f':
          if (isConst) {
            if (NDim==1) createAndFillNDarrayThenPutInEvent<const float,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) createAndFillNDarrayThenPutInEvent<const float,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) createAndFillNDarrayThenPutInEvent<const float,3>(shape, m_eventCounter, evt, key);
          } else {
            if (NDim==1) createAndFillNDarrayThenPutInEvent<float,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) createAndFillNDarrayThenPutInEvent<float,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) createAndFillNDarrayThenPutInEvent<float,3>(shape, m_eventCounter, evt, key);
          }
          break;
        case 'd':
          if (isConst) {
            if (NDim==1) createAndFillNDarrayThenPutInEvent<const double,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) createAndFillNDarrayThenPutInEvent<const double,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) createAndFillNDarrayThenPutInEvent<const double,3>(shape, m_eventCounter, evt, key);
          } else {
            if (NDim==1) createAndFillNDarrayThenPutInEvent<double,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) createAndFillNDarrayThenPutInEvent<double,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) createAndFillNDarrayThenPutInEvent<double,3>(shape, m_eventCounter, evt, key);
          }
          break;
        case 'i':
          if (isConst) {
            if (NDim==1) createAndFillNDarrayThenPutInEvent<const int,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) createAndFillNDarrayThenPutInEvent<const int,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) createAndFillNDarrayThenPutInEvent<const int,3>(shape, m_eventCounter, evt, key);
          } else {
            if (NDim==1) createAndFillNDarrayThenPutInEvent<int,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) createAndFillNDarrayThenPutInEvent<int,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) createAndFillNDarrayThenPutInEvent<int,3>(shape, m_eventCounter, evt, key);
          }
          break;
        case 'u':
          if (isConst) {
            if (NDim==1) createAndFillNDarrayThenPutInEvent<const unsigned int, 1>(shape, m_eventCounter, evt, key);
            if (NDim==2) createAndFillNDarrayThenPutInEvent<const unsigned int,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) createAndFillNDarrayThenPutInEvent<const unsigned int,3>(shape, m_eventCounter, evt, key);
          } else {
            if (NDim==1) createAndFillNDarrayThenPutInEvent<unsigned int,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) createAndFillNDarrayThenPutInEvent<unsigned int,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) createAndFillNDarrayThenPutInEvent<unsigned int,3>(shape, m_eventCounter, evt, key);
          }
          break;
        }
      }
    } // end adding all the arrays
  }
protected:
  template < class T, unsigned NDim>
  void createAndFillNDarrayThenPutInEvent(const unsigned *shape, size_t fillValue, Event &evt, const string &key) {
    enum ndns::Order order = ndns::C;
    if (m_use_fortran_stride) order = ndns::Fortran;
    unsigned numberElements = 1;
    for (unsigned idx = 0; idx < NDim; ++idx) numberElements *= shape[idx];
    typedef typename RemoveConst<T>::type NonConstT;
    boost::shared_ptr<NonConstT> data = boost::shared_ptr<NonConstT>(new NonConstT[numberElements]);
    NonConstT *p = data.get();
    for (unsigned idx = 0; idx < numberElements; ++idx) *p++ = NonConstT(fillValue + idx);
    boost::shared_ptr< ndarray<T,NDim> > arrayPtr = boost::make_shared< ndarray<T,NDim> >(data, shape, order);
    evt.put(arrayPtr, key);
    MsgLog(name(), debug,"  array: " << *arrayPtr);
  }

private:
  size_t m_eventCounter;
  bool m_use_fortran_stride, m_vary_array_sizes;
  bool m_vlen_prefix;
  string m_ndarrayKeyPrefix;
  int m_skipEvent;
};

PSANA_MODULE_FACTORY(TestModuleNDArrayString);


// ---------------------------------------------
// Reading modules:

// this class is a base for the two testing modules below, this class is not to be used directly
class TestModuleReadNDArrayString : public Module {
public:
  TestModuleReadNDArrayString(std::string moduleName, bool vlen) 
    : Module(moduleName), m_vlen(vlen) {}

  virtual void beginJob(Event& evt, Env& env) {
    m_eventCounter = 0;
  }

  virtual void event(Event& evt, Env& env) {
    ++m_eventCounter;
    MsgLog(name(),debug,"Event number (1-up counter): " << m_eventCounter);
    ostringstream expected_s1, expected_s2;
    expected_s1 << "This is event number: " << m_eventCounter;
    expected_s2 << "This is a second string.  10 * event number is " << 10*m_eventCounter;

    boost::shared_ptr<std::string> s1 = evt.get("my_string1");
    boost::shared_ptr<std::string> s2 = evt.get("my_string2");
    if (not s1) {
      MsgLog(name(),error,"std::string with key 'my_string1' not in event number (1-up counter): " 
             << m_eventCounter);
    } else if (*s1 != expected_s1.str()) {
      MsgLog(name(),error,"std::string with key 'my_string1' is not equal to expected."
             << " expected: '" << expected_s1.str() << "' while s1='" << *s1 << "'");
    }       
    if (not s2) {
      MsgLog(name(),error,"std::string with key 'my_string2' not in event number (1-up counter): " 
             << m_eventCounter);
    } else if (*s2 != expected_s2.str()) {
      MsgLog(name(),error,"std::string with key 'my_string2' is not equal to expected."
             << " expected: '" << expected_s2.str() << "' while s2='" << *s2 << "'");
    }

    // gett 10 nd arrays from event queue, const and non-const
    const unsigned DIM = 2;
    const unsigned DIM1 = std::min(20UL, DIM + (m_vlen ? m_eventCounter : 0));
    const unsigned DIM2 = DIM;
    const unsigned DIM3 = DIM;
    const unsigned twoDshape[2] = {DIM1,DIM2};
    const unsigned threeDshape[3] = {DIM1,DIM2,DIM3};
    const unsigned oneDshape[1] = {DIM1};
    
    const int numberOfNDarrays = 5;
    const char *keys[numberOfNDarrays] = {"my_float2Da", "my_float2Db", "my_double3D", "my_int1D", "my_uint1D"};
    const char types[numberOfNDarrays] = {'f', 'f', 'd', 'i', 'u'};
    const int NDims[numberOfNDarrays] = {2,2,3,1,1};
    const unsigned *shapes[numberOfNDarrays] = {twoDshape, twoDshape, threeDshape, oneDshape, oneDshape}; 
    
    for (int i = 0; i < numberOfNDarrays; ++i) {
      const char *_key = keys[i];
      const int NDim = NDims[i];
      const unsigned *shape = shapes[i];
      for (int isConst = 0; isConst < 2; ++isConst) {
        string key = string(_key);
        if (isConst) key = "c" + key;
        if (m_vlen) key = ndarrayVlenPrefix() + std::string(":") + key;
        MsgLog(name(), debug, "  array " << i << " const=" << isConst << " key=" << key);
        switch (types[i]) {
        case 'f':
          if (isConst) {
            if (NDim==1) testNDarrayInEvent<const float,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) testNDarrayInEvent<const float,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) testNDarrayInEvent<const float,3>(shape, m_eventCounter, evt, key);
          } else {
            if (NDim==1) testNDarrayInEvent<float,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) testNDarrayInEvent<float,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) testNDarrayInEvent<float,3>(shape, m_eventCounter, evt, key);
          }
          break;
        case 'd':
          if (isConst) {
            if (NDim==1) testNDarrayInEvent<const double,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) testNDarrayInEvent<const double,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) testNDarrayInEvent<const double,3>(shape, m_eventCounter, evt, key);
          } else {
            if (NDim==1) testNDarrayInEvent<double,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) testNDarrayInEvent<double,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) testNDarrayInEvent<double,3>(shape, m_eventCounter, evt, key);
          }
          break;
        case 'i':
          if (isConst) {
            if (NDim==1) testNDarrayInEvent<const int,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) testNDarrayInEvent<const int,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) testNDarrayInEvent<const int,3>(shape, m_eventCounter, evt, key);
          } else {
            if (NDim==1) testNDarrayInEvent<int,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) testNDarrayInEvent<int,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) testNDarrayInEvent<int,3>(shape, m_eventCounter, evt, key);
          }
          break;
        case 'u':
          if (isConst) {
            if (NDim==1) testNDarrayInEvent<const unsigned int, 1>(shape, m_eventCounter, evt, key);
            if (NDim==2) testNDarrayInEvent<const unsigned int,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) testNDarrayInEvent<const unsigned int,3>(shape, m_eventCounter, evt, key);
          } else {
            if (NDim==1) testNDarrayInEvent<unsigned int,1>(shape, m_eventCounter, evt, key);
            if (NDim==2) testNDarrayInEvent<unsigned int,2>(shape, m_eventCounter, evt, key);
            if (NDim==3) testNDarrayInEvent<unsigned int,3>(shape, m_eventCounter, evt, key);
          }
          break;
        }
      }
    } // end adding all the arrays
  }
protected:
  template < class T, unsigned NDim>
  void testNDarrayInEvent(const unsigned *expectedShape, size_t fillValue, Event &evt, const string &key) {
    unsigned numberElements = 1;
    for (unsigned idx = 0; idx < NDim; ++idx) numberElements *= expectedShape[idx];
    boost::shared_ptr< ndarray<T,NDim> > arrayPtr = evt.get(key);
    if (not arrayPtr) {
      MsgLog(name(), error, "array for key " << key << " not found");
      return;
    } 
    // test shape
    const unsigned *shape = arrayPtr->shape();
    for (unsigned shapeIdx = 0; shapeIdx < NDim; ++shapeIdx) {
      if (shape[shapeIdx] != expectedShape[shapeIdx]) {
        MsgLog(name(), error, "ndarray for key= " << key << " does not have expected dim: " 
               << expectedShape[shapeIdx] << " for dim = " << shapeIdx
               << " the value is " << shape[shapeIdx]);
      }
    }
    // test values
    T *p = arrayPtr->data();
    for (unsigned idx = 0; idx < numberElements; ++idx) {
      long double curValue = *p++;
      long double expectedValue = fillValue + idx;
      long double eps = 1e-6;
      if (std::abs(expectedValue) > eps) {
        MsgLog(name(), error, "array value for idx= " 
               << idx << " is wrong. expected: " << expectedValue
               << " but got " << curValue);
      }
    }
  }

private:
  bool m_vlen;
  size_t m_eventCounter;
};


class TestModuleReadVlenNDArrayString : public TestModuleReadNDArrayString {
public:
  TestModuleReadVlenNDArrayString(std::string moduleName) :
    TestModuleReadNDArrayString(moduleName, true) {}
};

PSANA_MODULE_FACTORY(TestModuleReadVlenNDArrayString);

class TestModuleReadNonVlenNDArrayString : public TestModuleReadNDArrayString {
public:
  TestModuleReadNonVlenNDArrayString(std::string moduleName) :
    TestModuleReadNDArrayString(moduleName, false) {}
};

PSANA_MODULE_FACTORY(TestModuleReadNonVlenNDArrayString);

} // namespace Translator
