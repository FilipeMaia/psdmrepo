#include <map>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/make_shared.hpp"

#include "hdf5/hdf5.h"

#include "MsgLogger/MsgLogger.h"
#include "PSEvt/TypeInfoUtils.h"

#include "Translator/NDArrayParams.h"

using namespace std;

using namespace Translator;

namespace {

const char * logger = "NDArrayParams";

typedef std::map<const std::type_info *, 
                 boost::shared_ptr<const NDArrayParameters>,
                 PSEvt::TypeInfoUtils::lessTypeInfoPtr> NDArrayParamMap;

NDArrayParamMap paramMap;


vector<string> remove_blanks(vector<string> args) {
  vector<string> removed;
  for (unsigned idx = 0; idx < args.size(); ++idx) {
    if (args.at(idx).size()>0) removed.push_back(args.at(idx));
  }
  return removed;
}

boost::shared_ptr<const NDArrayParameters> addNDArrayToParamMap(const std::type_info * ndarrayTypeInfoPtr) {
  string realName = PSEvt::TypeInfoUtils::typeInfoRealName(ndarrayTypeInfoPtr);
  static string ndarray("ndarray");
  boost::shared_ptr<const NDArrayParameters> nullPtr;

  string::size_type ndarrayPos = realName.find("ndarray");
  if (ndarrayPos == string::npos) {
    MsgLog(logger, error, "addNDArrayToParamMap: string 'ndarray' not present in type name " << realName);
    return nullPtr;
  }
  string after_ndarray = realName.substr(ndarrayPos + ndarray.size());
  vector<string> templateParams;
  boost::split(templateParams, after_ndarray, boost::is_any_of("<,>"), boost::token_compress_on);
  templateParams = remove_blanks(templateParams);
  if (templateParams.size() != 2) {
    MsgLog(logger, error, "addNDArrayToParamMap: after spliting templateParams = " 
           << after_ndarray << " on <,> had " << templateParams.size() 
           << " items instead of 2");
    return nullPtr;
  }
  boost::trim(templateParams.at(0));
  boost::trim(templateParams.at(1));
  string elemType = templateParams.at(0);
  string dimStr = templateParams.at(1);
  if (elemType.size()==0 or dimStr.size()==0) {
    MsgLog(logger, error, "addNDArrayToParamMap: one of trimmed templateParams is blank");
    return nullPtr;
  }
  if (dimStr.at(dimStr.size()-1)=='u') dimStr = dimStr.substr(0,dimStr.size()-1);
  unsigned dim = 0;
  try {
    dim = boost::lexical_cast<unsigned>(dimStr);
  } catch (const boost::bad_lexical_cast &) {
    MsgLog(logger, error, "addNDArrayToParamMap: could not convert dim: " 
           << dim << " into an unsigned");
    return nullPtr;
  }
  
  bool constElem = false;
  static string constStr("const");
  if (boost::algorithm::starts_with(elemType,constStr)) {
    elemType = elemType.substr(constStr.size());
    boost::trim(elemType);
    constElem = true;
  }
  if (boost::algorithm::ends_with(elemType,constStr)) {
    elemType = elemType.substr(0, elemType.size()-constStr.size());
    boost::trim(elemType);
    constElem = true;
  }

  unsigned sizeBytes = 0;
  string elemName;
  bool isInt = false;
  bool isFloat = false;
  bool isSigned = false;

  if (elemType == PSEvt::TypeInfoUtils::typeInfoRealName(&typeid(float))) {
    sizeBytes = sizeof(float);
    elemName = "float";
    isFloat = true;
  } else if (elemType == PSEvt::TypeInfoUtils::typeInfoRealName(&typeid(double))) {
    sizeBytes = sizeof(double);
    elemName = "float";
    isFloat = true;
  } else if (elemType == PSEvt::TypeInfoUtils::typeInfoRealName(&typeid(long double))) {
    sizeBytes = sizeof(long double);
    elemName = "float";
    isFloat = true;
  } else if (elemType == PSEvt::TypeInfoUtils::typeInfoRealName(&typeid(int8_t))) {
    sizeBytes = sizeof(int8_t);
    elemName = "int";
    isInt = true;
    isSigned = true;
  } else if (elemType == PSEvt::TypeInfoUtils::typeInfoRealName(&typeid(uint8_t))) {
    sizeBytes = sizeof(uint8_t);
    elemName = "uint";
    isInt = true;
  } else if (elemType == PSEvt::TypeInfoUtils::typeInfoRealName(&typeid(int16_t))) {
    sizeBytes = sizeof(int16_t);
    elemName = "int";
    isInt = true;
    isSigned = true;
  } else if (elemType == PSEvt::TypeInfoUtils::typeInfoRealName(&typeid(uint16_t))) {
    sizeBytes = sizeof(uint16_t);
    elemName = "uint";
    isInt = true;
  } else if (elemType == PSEvt::TypeInfoUtils::typeInfoRealName(&typeid(int32_t))) {
    sizeBytes = sizeof(int32_t);
    elemName = "int";
    isInt = true;
    isSigned = true;
  } else if (elemType == PSEvt::TypeInfoUtils::typeInfoRealName(&typeid(uint32_t))) {
    sizeBytes = sizeof(uint32_t);
    elemName = "uint";
    isInt = true;
  } else if (elemType == PSEvt::TypeInfoUtils::typeInfoRealName(&typeid(int64_t))) {
    sizeBytes = sizeof(int64_t);
    elemName = "int";
    isInt = true;
    isSigned = true;
  } else if (elemType == PSEvt::TypeInfoUtils::typeInfoRealName(&typeid(uint64_t))) {
    sizeBytes = sizeof(uint64_t);
    elemName = "uint";
    isInt = true;
  } else {
    MsgLog(logger, error, "addNDArrayToParamMap: elemType = " << elemType 
           << " is not equal to known base type");
    return nullPtr;
  }

  enum NDArrayParameters::ElemType elemTypeEnum = NDArrayParameters::unknownElemType;
  if (isInt and isSigned) elemTypeEnum = NDArrayParameters::intElemType;
  else if (isInt and not isSigned) elemTypeEnum = NDArrayParameters::uintElemType;
  else if (isFloat) elemTypeEnum = NDArrayParameters::floatElemType;
  paramMap[ndarrayTypeInfoPtr] = 
    boost::make_shared<NDArrayParameters>(elemName, elemTypeEnum, 
                                          sizeBytes, dim, constElem);
  return paramMap[ndarrayTypeInfoPtr];
}

}; // local namespace

NDArrayParameters::NDArrayParameters()
  : m_elemName(""), m_elemType(unknownElemType),
    m_sizeBytes(0), m_dim(0), m_isConstElem(false) 
{
}

NDArrayParameters::NDArrayParameters(std::string elemName, ElemType elemType,
                                     unsigned sizeBytes, unsigned dim, bool isConstElem) 
  : m_elemName(elemName), m_elemType(elemType),
    m_sizeBytes(sizeBytes), m_dim(dim), m_isConstElem(isConstElem) 
{
}

boost::shared_ptr<const NDArrayParameters> 
Translator::ndarrayParameters(const std::type_info *ndarrayTypeInfoPtr) {
  NDArrayParamMap::iterator pos;
  pos = ::paramMap.find(ndarrayTypeInfoPtr);
  if (pos == ::paramMap.end()) {
    return ::addNDArrayToParamMap(ndarrayTypeInfoPtr);
  }
  return pos->second;
}

string Translator::ndarrayGroupName(const std::type_info *ndarrayTypeInfoPtr) {
  NDArrayParamMap::iterator pos;
  pos = ::paramMap.find(ndarrayTypeInfoPtr);
  if (pos == ::paramMap.end()) {
    boost::shared_ptr<const NDArrayParameters> ptr = ::addNDArrayToParamMap(ndarrayTypeInfoPtr);
    if (not ptr) return "";
  }
  pos = ::paramMap.find(ndarrayTypeInfoPtr);
  if (pos == ::paramMap.end()) {
    MsgLog(logger, error, "unexpected - successfully added " 
           << PSEvt::TypeInfoUtils::typeInfoRealName(ndarrayTypeInfoPtr)
           << " but not found");
    return "";
  }
  const NDArrayParameters &params = *(pos->second);
  string groupName = "ndarray";  
  if (params.isConstElem()) { groupName += "_const"; }
  groupName += "_" + params.elemName();
  groupName += boost::lexical_cast<string>(8*params.sizeBytes());
  groupName += "_";
  groupName += boost::lexical_cast<string>(params.dim());
  return groupName;
}

