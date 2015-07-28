#include <map>
#include <vector>

#include "boost/algorithm/string.hpp"
#include "boost/lexical_cast.hpp"
#include "boost/make_shared.hpp"

#include "hdf5/hdf5.h"

#include "MsgLogger/MsgLogger.h"
#include "PSEvt/TypeInfoUtils.h"

#include "Translator/NDArrayUtil.h"

using namespace std;

using namespace Translator;

// local namespace - implementation
namespace {

const char * logger = "NDArrayUtil";

typedef std::map< const std::type_info *, 
                  boost::shared_ptr<const psddl_hdf2psana::NDArrayParameters> ,
                  PSEvt::TypeInfoUtils::lessTypeInfoPtr> NDArrayParamMap;

NDArrayParamMap paramMapVlen, paramMapNotVlen;

boost::shared_ptr<const psddl_hdf2psana::NDArrayParameters> find(const std::type_info *typeInfoPtr, 
                                                                 enum psddl_hdf2psana::NDArrayParameters::VlenDim vlenDim)
{
  NDArrayParamMap::iterator pos;
  bool found = false;
  switch (vlenDim) {
  case psddl_hdf2psana::NDArrayParameters::SlowDimNotVlen:
    pos = paramMapNotVlen.find(typeInfoPtr);
    found = (pos != paramMapNotVlen.end());
    break;
  case psddl_hdf2psana::NDArrayParameters::SlowDimIsVlen:
    pos = paramMapVlen.find(typeInfoPtr);
    found = (pos != ::paramMapVlen.end());
    break;
  }
  if (found) return pos->second;;
  boost::shared_ptr<const psddl_hdf2psana::NDArrayParameters> nullPtr;
  return nullPtr;
}

vector<string> remove_blanks(vector<string> args) {
  vector<string> removed;
  for (unsigned idx = 0; idx < args.size(); ++idx) {
    if (args.at(idx).size()>0) removed.push_back(args.at(idx));
  }
  return removed;
}

boost::shared_ptr<const psddl_hdf2psana::NDArrayParameters> 
addNDArrayToParamMap(const std::type_info * ndarrayTypeInfoPtr,
                     enum psddl_hdf2psana::NDArrayParameters::VlenDim vlenDim) {

  string realName = PSEvt::TypeInfoUtils::typeInfoRealName(ndarrayTypeInfoPtr);
  static string ndarray("ndarray");
  boost::shared_ptr<const psddl_hdf2psana::NDArrayParameters> nullPtr;

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

  enum psddl_hdf2psana::NDArrayParameters::ElemType elemTypeEnum = 
    psddl_hdf2psana::NDArrayParameters::unknownElemType;
  if (isInt and isSigned) elemTypeEnum = psddl_hdf2psana::NDArrayParameters::intElemType;
  else if (isInt and not isSigned) elemTypeEnum = psddl_hdf2psana::NDArrayParameters::uintElemType;
  else if (isFloat) elemTypeEnum = psddl_hdf2psana::NDArrayParameters::floatElemType;

  boost::shared_ptr<psddl_hdf2psana::NDArrayParameters> params = 
    boost::make_shared<psddl_hdf2psana::NDArrayParameters>(elemName, elemTypeEnum, 
                                                           sizeBytes, dim, constElem, vlenDim);
  switch (vlenDim) {
  case psddl_hdf2psana::NDArrayParameters::SlowDimNotVlen:
    paramMapNotVlen[ndarrayTypeInfoPtr] = params;
    break;
  case psddl_hdf2psana::NDArrayParameters::SlowDimIsVlen:
    paramMapVlen[ndarrayTypeInfoPtr] = params;
    break;
  }
  
  return params;
}

}; // local namespace


boost::shared_ptr<const psddl_hdf2psana::NDArrayParameters> 
Translator::ndarrayParameters(const std::type_info *ndarrayTypeInfoPtr,
                     enum psddl_hdf2psana::NDArrayParameters::VlenDim vlenDim) 
{
  // check cache
  boost::shared_ptr<const psddl_hdf2psana::NDArrayParameters> params = 
    find(ndarrayTypeInfoPtr, vlenDim);
  
  if (not params) {
    return addNDArrayToParamMap(ndarrayTypeInfoPtr, vlenDim);
  }
  return params;
}

string Translator::ndarrayGroupName(const std::type_info *ndarrayTypeInfoPtr,
                     enum psddl_hdf2psana::NDArrayParameters::VlenDim vlenDim) 
{
  // check cache
  boost::shared_ptr<const psddl_hdf2psana::NDArrayParameters> params = 
    find(ndarrayTypeInfoPtr, vlenDim);
  
  if (not params) {
    params = addNDArrayToParamMap(ndarrayTypeInfoPtr, vlenDim);
  }
  if (not params) return "";

  string groupName = "ndarray";  
  if (params->isConstElem()) { groupName += "_const"; }
  groupName += "_" + params->elemName();
  groupName += boost::lexical_cast<string>(8*(params->sizeBytes()));
  groupName += "_";
  groupName += boost::lexical_cast<string>(params->dim());
  if (params->isVlen()) groupName += "_vlen";
  return groupName;
}

