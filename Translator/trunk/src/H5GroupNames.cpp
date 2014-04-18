#include "Translator/H5GroupNames.h"
#include "Translator/doNotTranslate.h"
#include "Translator/NDArrayParams.h"

#include <string>
#include <sstream>

#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/BldInfo.hh"
#include "pdsdata/xtc/ProcInfo.hh"
#include "PSEvt/TypeInfoUtils.h"
#include "PSEvt/EventKey.h"
#include "MsgLogger/MsgLogger.h"

using namespace Translator;
using namespace std;

namespace {

  const char * logger = "H5GroupNames";

  const std::string srcKeySepStr = "__";

  std::string strDetInfo(const Pds::DetInfo& src, bool detInfoSpecialAsAstrerik)
  {
    std::ostringstream str ;
    if (src.detector() == 0xff) {
      str << (detInfoSpecialAsAstrerik ? "*" : "detector");
    } else {
      str << Pds::DetInfo::name(src.detector());
    }
    str << '.';
    if (src.detId() == 0xff) {
      str << (detInfoSpecialAsAstrerik ? "*" : "id");
    } else {
      str << src.detId();
    }
    str << ':';
    if (src.device() == 0xff) {
      str << (detInfoSpecialAsAstrerik ? "*" : "device");
    } else {
      str << Pds::DetInfo::name(src.device());
    }
    str << '.';
    if (src.devId() == 0xff) {
      str << (detInfoSpecialAsAstrerik ? "*" : "id");
    } else {
      str << src.devId();
    }
    return str.str();
  }

  std::string strBldInfo(const Pds::BldInfo& src)
  {
    std::ostringstream str;
    if (unsigned(src.type()) != 0xffffffff) str << Pds::BldInfo::name(src);
    return str.str();
  }

  std::string strProcInfo(const Pds::ProcInfo& src, bool pidSpaceSepAtEnd)
  {
    std::ostringstream str;
    uint32_t ip = src.ipAddr();
    if (not pidSpaceSepAtEnd) str << src.processId() << '@';
    str << ((ip>>24)&0xff) 
        << '.' << ((ip>>16)&0xff)
        << '.' << ((ip>>8)&0xff) 
        << '.' << (ip&0xff);
    if (pidSpaceSepAtEnd) str << ", pid=" << src.processId();
    return str.str();
  }

  std::string srcName(const Pds::Src& src, bool procPidSpaceSepAtEnd, bool detInfoSpecialAsAstrerik)
  {
    if ((src == PSEvt::EventKey::noSource()) or (src == PSEvt::EventKey::anySource())) {
      return std::string("noSrc");
    } else if (src.level() == Pds::Level::Source) {
      return ::strDetInfo(static_cast<const Pds::DetInfo&>(src),detInfoSpecialAsAstrerik);
    } else if (src.level() == Pds::Level::Reporter) {
      return ::strBldInfo(static_cast<const Pds::BldInfo&>(src));
    } else if (src.level() == Pds::Level::Control) {
      return std::string("Control");
    } else if (src.level() == Pds::Level::Event) {
      return std::string("Event");
    }
    return ::strProcInfo(static_cast<const Pds::ProcInfo&>(src),procPidSpaceSepAtEnd);
  }

  std::string replaceCharactersThatAreBadForH5GroupNames(const std::string & orig) {
    std::string newString;
    for (unsigned idx = 0; idx < orig.size(); ++idx) {
      if (orig[idx] == '/') {
        newString.push_back('_');
      } else {
        newString.push_back(orig[idx]);
      }
    }
    return newString;
  }

} // local namespace

namespace Translator {

H5GroupNames::H5GroupNames(const std::string & calibratedKey, const TypeAliases::TypeInfoSet & ndarrays) 
  : m_calibratedKey(calibratedKey), m_ndarrays(ndarrays) 
{}

string H5GroupNames::nameForType(const std::type_info *typeInfoPtr) {
  if (m_ndarrays.find(typeInfoPtr) != m_ndarrays.end()) {
    string groupName = ndarrayGroupName(typeInfoPtr);
    if (groupName.size()>0) return groupName;
    string realName = PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtr);
    MsgLog(logger, error, "type " << realName << " matches known NDArrays "
           << "but ndarrayGroupName() returned empty name - NOT simplifying name.");
  }
  string realName = PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtr);
  string::size_type leftParenIdx = realName.find("(");
  if (leftParenIdx != string::npos) {
    realName = realName.substr(0,leftParenIdx);
  }
  // strip Psana:: from the front
  static const string psana("Psana::");
  if (realName.size() > psana.size() and realName.substr(0,psana.size()) == psana) {
    realName = realName.substr(psana.size());
  }
  
  // replace CsPad::DataV with CsPad::ElementV for backward compatibility
  static const string csPadDataV("CsPad::DataV");
  static const string csPadElementV("CsPad::ElementV");
  if ((realName.size()>csPadDataV.size()) and (realName.substr(0,csPadDataV.size()) == csPadDataV)) {
    realName = csPadElementV + realName.substr(csPadDataV.size());
  }

  // replace PNCCD::FramesV with PNCCD::FrameV for backward compatibility
  static const string PNCCDFrameV("PNCCD::FrameV");
  static const string PNCCDFramesV("PNCCD::FramesV");
  if ((realName.size()>PNCCDFramesV.size()) and (realName.substr(0,PNCCDFramesV.size()) == PNCCDFramesV)) {
    realName = PNCCDFrameV  + realName.substr(PNCCDFramesV.size());
  }

  // replace Acqiris::TdcConfigV with Acqiris::AcqirisTdcConfigV for backward compatibility
  static const string AcqirisTdcConfigV("Acqiris::TdcConfigV");
  static const string AcqirisAcqirisTdcConfigV("Acqiris::AcqirisTdcConfigV");
  if ((realName.size()>AcqirisTdcConfigV.size()) and (realName.substr(0,AcqirisTdcConfigV.size()) == AcqirisTdcConfigV)) {
    realName = AcqirisAcqirisTdcConfigV  + realName.substr(AcqirisTdcConfigV.size());
  }

  return realName;
}
  
void H5GroupNames::addTypeAttributes(hdf5pp::Group group, const std::type_info *typeInfoPtr) 
{
  if (isNDArray(typeInfoPtr)) {
    group.createAttr<uint8_t>("_ndarray").store(1);
    boost::shared_ptr<const NDArrayParameters> params = ndarrayParameters(typeInfoPtr);
    if (not params) {
      MsgLog(logger,error, "addTypeAttributes: group=" << group.name()
             << " type=" << PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtr)
             << " no ndarrayParams found");
      group.createAttr<uint32_t>("_ndarrayDim").store(0);
      group.createAttr<int16_t>("_ndarrayElemType").store( 
                                int16_t(NDArrayParameters::unknownElemType));
      group.createAttr<uint32_t>("_ndarraySizeBytes").store(0);
      group.createAttr<uint8_t>("_ndarrayConstElem").store(0);
    } else {
      group.createAttr<uint32_t>("_ndarrayDim").store(params->dim());
      group.createAttr<int16_t>("_ndarrayElemType").store(uint16_t(params->elemType()));
      group.createAttr<uint32_t>("_ndarraySizeBytes").store(params->sizeBytes());
      group.createAttr<uint8_t>("_ndarrayConstElem").store(uint8_t(params->isConstElem()));
    }
  } else {
    // not an ndarray
    group.createAttr<uint8_t>("_ndarray").store(0);
  }
}

string H5GroupNames::nameForSrc(const Pds::Src &src) {
  return ::srcName(src,false,false);
}

std::pair<std::string, std::string> H5GroupNames::nameForSrcKey(const Pds::Src &src, 
                                                                const std::string &key) {
  string srcKeyGroupName = nameForSrc(src); // should always be non-zero length
  if (key == m_calibratedKey) return pair<string,string>(srcKeyGroupName,string());
  string keyStringToAddOrig;
  hasDoNotTranslatePrefix(key,&keyStringToAddOrig);
  string keyStringToAdd = replaceCharactersThatAreBadForH5GroupNames(keyStringToAddOrig);
  if (keyStringToAdd.size()>0) {
    srcKeyGroupName += (srcKeySep() + keyStringToAdd);
  }
  return pair<string,string>(srcKeyGroupName,keyStringToAdd);
}


std::string H5GroupNames::srcKeySep() {
  return ::srcKeySepStr;
}

} // namespace Translator
