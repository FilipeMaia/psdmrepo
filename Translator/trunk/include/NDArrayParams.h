#ifndef TRANSLATOR_NDARRAYPARAMS_H
#define TRANSLATOR_NDARRAYPARAMS_H

#include <string>
#include <typeinfo>
#include "boost/shared_ptr.hpp"

namespace Translator {

class NDArrayParameters {
 public:
  enum ElemType { unknownElemType=0, intElemType=1, uintElemType=2, floatElemType=3}; 
  std::string elemName() const { return m_elemName; }
  ElemType elemType() const { return m_elemType; }
  unsigned sizeBytes() const { return m_sizeBytes; }
  unsigned dim() const { return m_dim; }
  bool isConstElem() const { return m_isConstElem; }
  NDArrayParameters(); 
  NDArrayParameters(std::string elemName, ElemType elemType, 
                    unsigned sizeBytes, unsigned dim, bool isConstElem);
 private:
  std::string m_elemName;
  ElemType m_elemType;
  unsigned m_sizeBytes;
  unsigned m_dim;
  bool m_isConstElem;
};

boost::shared_ptr<const NDArrayParameters>  ndarrayParameters(const std::type_info *ndarrayTypeInfoPtr);

std::string ndarrayGroupName(const std::type_info *ndarrayTypeInfoPtr, bool vlen = false);

} // namespace Translator

#endif
