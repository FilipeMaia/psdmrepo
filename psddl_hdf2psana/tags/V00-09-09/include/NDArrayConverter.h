#ifndef PSDDL_HDF2PSANA_NDARRAYCONVERTER_H
#define PSDDL_HDF2PSANA_NDARRAYCONVERTER_H

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "pdsdata/xtc/Src.hh"
#include "PSEvt/Event.h"

namespace psddl_hdf2psana {

// forward declaration of class that describes ndarray to be converted
class NDArrayParameters;

// class that does conversion given ndarray parameters
class NDArrayConverter {
 public: 
  NDArrayConverter() {};
  void convert(const hdf5pp::Group& group, int64_t idx, 
               const NDArrayParameters & ndArrayParams, 
               int schema_version, const Pds::Src &src, 
               const std::string &key, PSEvt::Event &evt) const;
};

// data required to convert
class NDArrayParameters {
 public:
  enum ElemType { unknownElemType=0, intElemType=1, uintElemType=2, floatElemType=3}; 
  enum  VlenDim {SlowDimNotVlen=0, SlowDimIsVlen=1};
  NDArrayParameters(std::string elemName, ElemType elemType, 
                    unsigned sizeBytes, unsigned dim, bool isConstElem, VlenDim vlenDim) 
    : m_elemName(elemName), m_elemType(elemType), m_sizeBytes(sizeBytes),
      m_dim(dim), m_isConstElem(isConstElem), m_isVlen(bool(vlenDim)) {}

  NDArrayParameters() {}

  // getters
  ElemType elemType() const { return m_elemType; }
  std::string elemName() const { return m_elemName; }
  unsigned sizeBytes() const { return m_sizeBytes; }
  unsigned dim() const { return m_dim; }
  bool isConstElem() const { return m_isConstElem; }
  bool isVlen() const { return m_isVlen; }

  // setters
  void  elemType(const ElemType _elemType) { m_elemType = _elemType; }
  void  elemName(const std::string & _elemName) { m_elemName = _elemName; }
  void  sizeBytes(const unsigned _sizeBytes) { m_sizeBytes  = _sizeBytes; }
  void dim(const unsigned _dim) { m_dim = _dim; }
  void isConstElem(const bool _isConstElem) { m_isConstElem = _isConstElem; }
  void isVlen(const bool _isVlen) {  m_isVlen = _isVlen; }

 private:
  std::string m_elemName;
  ElemType m_elemType;
  unsigned m_sizeBytes;
  unsigned m_dim;
  bool m_isConstElem;
  bool m_isVlen;
};

}  // namespace psddl_hdf2psana

#endif
