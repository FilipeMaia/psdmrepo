#include "Translator/H5GroupNames.h"
#include "Translator/doNotTranslate.h"

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

  /**
   * srcName and the functions strDetInfo nand  strBldInfo above are just a slight variation
   * of the code in PSEvt::EventKey.cpp.  It provides flags to print the srcName the way that 
   * PSEvt::srcName (in EventKey.cpp) does, as well as they they will be used for hdf5 
   * translation.  
   * Calling the function as 
   *     srcName(src,true,true)
   * mimics PSEvt::srcName, and calling the function as
   *     srcName(src,false,false)
   * gives the src names for translation.
   * when the flags are true, you can get a space ' ', or an asterick '*' in the 
   * src name, which we do not want for translation. 
   */
  std::string srcName(const Pds::Src& src, bool procPidSpaceSepAtEnd, bool detInfoSpecialAsAstrerik)
  {
    if ((src == PSEvt::EventKey::anySource()) or (src == PSEvt::EventKey::noSource())) return "";
    if (src.level() == Pds::Level::Source) {
      return ::strDetInfo(static_cast<const Pds::DetInfo&>(src),detInfoSpecialAsAstrerik);
    } else if (src.level() == Pds::Level::Reporter) {
      return ::strBldInfo(static_cast<const Pds::BldInfo&>(src));
    } else if (src.level() == Pds::Level::Control) {
      return std::string("Control");
    } else if (src.level() == Pds::Level::NumberOfLevels) {
      // special match-anything source, empty string
      return std::string();
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

H5GroupNames::H5GroupNames(bool short_bld_name, const TypeAliases::TypeInfoSet & ndarrays) 
  : m_short_bld_name(short_bld_name),
    m_ndarrays(ndarrays) 
{}

string H5GroupNames::nameForType(const std::type_info *typeInfoPtr) {
  if (m_ndarrays.find(typeInfoPtr) != m_ndarrays.end()) return "NDArray";
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
  
  // shorten Bld::Bld to Bld:: if option is set
  static const string BldBld("Bld::Bld");
  static const string Bld("Bld::");
  if (m_short_bld_name) {
    if (realName.size() > BldBld.size() and realName.substr(0,BldBld.size()) == BldBld) {
      realName = realName.substr(Bld.size());
    }
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
  
string H5GroupNames::nameForSrc(const Pds::Src &src) {
  return ::srcName(src,false,false);
}

std::string H5GroupNames::nameForSrcKey(const Pds::Src &src, const std::string &key) {
  string srcKeyGroupName = nameForSrc(src);
  string keyStringToAddOrig;
  hasDoNotTranslatePrefix(key,&keyStringToAddOrig);
  string keyStringToAdd = replaceCharactersThatAreBadForH5GroupNames(keyStringToAddOrig);
  if (keyStringToAdd.size()>0) {
    if (srcKeyGroupName.size()>0) srcKeyGroupName += "_";
    srcKeyGroupName += keyStringToAdd;
  }
  if (srcKeyGroupName.size()==0) {
    srcKeyGroupName = "no_src";
  }
  return srcKeyGroupName;
}

} // namespace Translator
