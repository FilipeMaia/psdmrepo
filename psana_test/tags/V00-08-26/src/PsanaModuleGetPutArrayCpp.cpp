#include "psana/Module.h"
#include "ndarray/ndarray.h"

template<class T>
class ArrayDelete {
public:
  void operator()(T*p) { delete[] p; };
};

/* ---------------------------
 * Puts a const and nonconst ndarray in the event store. 
 * Both arrays are 3 x 4 2D arrays of float with the values 
 * 0,1,2,...11
 * 
 * set     const_key to be the key for the const ndarray
 *      nonconst_key to be the array for the array for the nonconst ndarray
 * ---------------------------- */
class PsanaModulePutNDArrayCpp : public Module {
public:
  std::string m_constKey, m_nonConstKey;
  PsanaModulePutNDArrayCpp(std::string moduleName) : Module(moduleName) 
  {
    m_constKey = configStr("const_key", "");
    m_nonConstKey = configStr("nonconst_key", "");
  }

  virtual void event(Event &evt, Env &env) {
    const unsigned NELEM = 12;
    const unsigned dim[] = {3,4};
    boost::shared_ptr<float> constData(new float[NELEM],ArrayDelete<float>());
    boost::shared_ptr<float> nonconstData(new float[NELEM],ArrayDelete<float>());
    for (unsigned k = 0; k < NELEM; ++k) {
      *(constData.get()+k) = float(k);
      *(nonconstData.get()+k) = float(k);
    }
    boost::shared_ptr< ndarray<float,2> > nonConstArr = 
      boost::make_shared< ndarray<float,2> >(nonconstData,dim);
    boost::shared_ptr< ndarray<const float,2> > constArr = 
      boost::make_shared< ndarray<const float,2> >(constData,dim);
    if (m_nonConstKey.size()) evt.put(nonConstArr,m_nonConstKey);
    if (m_constKey.size()) evt.put(constArr,m_constKey);
  }
};

/* ---------------------------
 * Gets a const and nonconst ndarray in the event store. 
 * 
 * set     const_key to be the key for the const ndarray
 *      nonconst_key to be the array for the array for the nonconst ndarray
 * if it gets the arrays from the event store it prints
 *    const_arr: #memory address#
 * nonconst_arr: #memory address#
 *
 * if it does not get them, then it prints the messages
 *    const_arr is null
 * nonconst_arr is null
 * ---------------------------- */
class PsanaModuleGetNDArrayCpp : public Module {
public:
  std::string m_constKey, m_nonConstKey;
  PsanaModuleGetNDArrayCpp(std::string moduleName) : Module(moduleName) 
  {
    m_constKey = configStr("const_key", "");
    m_nonConstKey = configStr("nonconst_key", "");
  }
  virtual void event(Event &evt, Env &env) {
    if (m_constKey.size()) {
      boost::shared_ptr< ndarray<const float,2> > const_arr = evt.get(m_constKey);
      if (const_arr) {
        MsgLog(name(),info,"const_arr: " << const_arr);
      } else {
        MsgLog(name(),info,"const_arr is null");
      }
    }
    if (m_nonConstKey.size()) {
      boost::shared_ptr< ndarray<float,2> > nonconst_arr = evt.get(m_nonConstKey);
      if (nonconst_arr) {
        MsgLog(name(), info, "nonconst_arr: " << nonconst_arr);
      } else {
        MsgLog(name(),info,"nonconst_arr is null");
      }
    }
  }
};

PSANA_MODULE_FACTORY(PsanaModulePutNDArrayCpp);
PSANA_MODULE_FACTORY(PsanaModuleGetNDArrayCpp);
