#include "psddl_hdf2psana/NDArrayConverter.h"
#include "psddl_hdf2psana/SchemaConstants.h"
#include "Translator/NDArrayUtil.h"

#include "Translator/H5GroupNames.h"
#include "Translator/specialKeyStrings.h"

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

string H5GroupNames::nameForType(const std::type_info *typeInfoPtr, const std::string &key) {
  if (m_ndarrays.find(typeInfoPtr) != m_ndarrays.end()) {
    enum psddl_hdf2psana::NDArrayParameters::VlenDim vlenDim = 
      psddl_hdf2psana::NDArrayParameters::SlowDimNotVlen;
    if ((key.size()>0) and hasNDArrayVlenPrefix(key)) {
      vlenDim = psddl_hdf2psana::NDArrayParameters::SlowDimIsVlen;
    }
    string groupName = ndarrayGroupName(typeInfoPtr, vlenDim);
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
  
void H5GroupNames::addTypeAttributes(hdf5pp::Group group, const std::type_info *typeInfoPtr, const std::string & key) 
{
  if (isNDArray(typeInfoPtr)) {
    group.createAttr<uint8_t>(psddl_hdf2psana::ndarrayAttrName).store(1);
    enum psddl_hdf2psana::NDArrayParameters::VlenDim vlenDim = 
      psddl_hdf2psana::NDArrayParameters::SlowDimNotVlen;
    if (hasNDArrayVlenPrefix(key)) {
      vlenDim = psddl_hdf2psana::NDArrayParameters::SlowDimIsVlen;
    }
    boost::shared_ptr<const psddl_hdf2psana::NDArrayParameters> params = 
      ndarrayParameters(typeInfoPtr, vlenDim);
    if (not params) {
      // addTypeAttributes is supposed to be called after the ndarray parameters
      // have been stored. If params is null, this is not the case, or the 
      // parameters were not deduced from this type.
      MsgLog(logger,warning, "addTypeAttributes: group=" << group.name()
             << " type=" << PSEvt::TypeInfoUtils::typeInfoRealName(typeInfoPtr)
             << " is ndarray, but no ndarrayParams found.");
      group.createAttr<uint32_t>(psddl_hdf2psana::ndarrayDimAttrName).store(0);
      group.createAttr<int16_t>(psddl_hdf2psana::ndarrayElemTypeAttrName).store( 
                                int16_t(psddl_hdf2psana::NDArrayParameters::unknownElemType));
      group.createAttr<uint32_t>(psddl_hdf2psana::ndarraySizeBytesAttrName).store(0);
      group.createAttr<uint8_t>(psddl_hdf2psana::ndarrayConstElemAttrName).store(0);
      group.createAttr<uint8_t>(psddl_hdf2psana::vlenAttrName).store(0);
    } else {
      group.createAttr<uint32_t>(psddl_hdf2psana::ndarrayDimAttrName).store(params->dim());
      group.createAttr<int16_t>(psddl_hdf2psana::ndarrayElemTypeAttrName).store(uint16_t(params->elemType()));
      group.createAttr<uint32_t>(psddl_hdf2psana::ndarraySizeBytesAttrName).store(params->sizeBytes());
      group.createAttr<uint8_t>(psddl_hdf2psana::ndarrayConstElemAttrName).store(uint8_t(params->isConstElem()));
      group.createAttr<uint8_t>(psddl_hdf2psana::vlenAttrName).store(uint8_t(params->isVlen()));
    }
  } else {
    // not an ndarray
    group.createAttr<uint8_t>(psddl_hdf2psana::ndarrayAttrName).store(0);
  }
}

string H5GroupNames::nameForSrc(const Pds::Src &src) {
  return ::srcName(src,false,false);
}

std::pair<std::string, std::string> H5GroupNames::nameForSrcKey(const Pds::Src &src, 
                                                                const std::string &key) {
  string srcKeyGroupName = nameForSrc(src); // should always be non-zero length
  if (key == m_calibratedKey) return pair<string,string>(srcKeyGroupName,string());
  string keyWithSpecialPrefixesStripped;
  bool doNotTranslate = hasDoNotTranslatePrefix(key,&keyWithSpecialPrefixesStripped);
  if (not doNotTranslate){
    hasNDArrayVlenPrefix(key, &keyWithSpecialPrefixesStripped);
  }
  string keyStringToAdd = replaceCharactersThatAreBadForH5GroupNames(keyWithSpecialPrefixesStripped);
  if (keyWithSpecialPrefixesStripped.size()>0) {
    srcKeyGroupName += (psddl_hdf2psana::srcEventKeySeperator + keyStringToAdd);
  }
  return pair<string,string>(srcKeyGroupName,keyStringToAdd);
}


} // namespace Translator
