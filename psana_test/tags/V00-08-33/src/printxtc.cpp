#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include "pdsdata/xtc/Xtc.hh"
#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/BldInfo.hh"
#include "pdsdata/xtc/ProcInfo.hh"
#include "pdsdata/psddl/bld.ddl.h"
#include "pdsdata/psddl/ipimb.ddl.h"
#include "pdsdata/psddl/lusi.ddl.h"
#include "pdsdata/psddl/epics.ddl.h"
#include "pdsdata/psddl/alias.ddl.h"

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

  std::string srcName(const Pds::Src& src)
  {
    bool procPidSpaceSepAtEnd = false;
    bool detInfoSpecialAsAstrerik = true;
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

} // local namespace

namespace psana_test {
uint32_t printXtcHeader(Pds::Xtc *xtc) {
  Pds::Damage &damage = xtc->damage;
  Pds::Src &src = xtc->src;
  Pds::TypeId &typeId = xtc->contains;
  uint32_t extent = xtc->extent;
  fprintf(stdout,"extent=%8.8X dmg=%5.5X src=%8.8X,%8.8X level=%d srcnm=%20s typeid=%2d ver=%d value=%5.5X compr=%d compr_ver=%d type_name=%s",
         extent,
         damage.value(),src.log(),src.phy(),src.level(), srcName(src).c_str(),
         typeId.id(), typeId.version(),typeId.value(),typeId.compressed(),typeId.compressed_version(),
         typeId.name(typeId.id()));
  return xtc->sizeofPayload();
}

namespace {

  const char *  sequenceType2string(enum Pds::Sequence::Type tp) {
    static const char *type2string[] = { "Event",
                                   "Occur",
                                   "Markr"};
    if ((tp>=0) and (size_t(tp) < sizeof(type2string)/sizeof(char *))) return type2string[int(tp)];
    return "*ERR*";
  }
} // local namepsace
  
uint32_t printXtcWithOffsetAndDepth(Pds::Xtc *xtc,int offset, int depth) {
  fprintf(stdout," xtc: off=%6ld dpth=%d ",(long int)offset,(int)depth);
  psana_test::printXtcHeader(xtc);
  fprintf(stdout,"\n");
  return xtc->extent;
}

Pds::Xtc * printTranslatedDgramHeader(Pds::Dgram *dgram) {
  Pds::Sequence & seq = dgram->seq;
  Pds::Env & env = dgram->env;
  Pds::Xtc * xtc = & dgram->xtc;
  const Pds::ClockTime & clock = seq.clock();
  const Pds::TimeStamp & stamp = seq.stamp();
  // The longest translated name for a TransitionId is 15 characters - BeginCalibCycle
  fprintf(stdout,"tp=%s sv=%15s ex=%d ev=%d sec=%8.8X nano=%8.8X tcks=%7.7X fid=%5.5X ctrl=%2.2X vec=%4.4X env=%8.8X",
         sequenceType2string(seq.type()),
         Pds::TransitionId::name(seq.service()),
         seq.isExtended(),seq.isEvent(),
         clock.seconds(),clock.nanoseconds(),
         stamp.ticks(), stamp.fiducials(),stamp.control(), stamp.vector(),env.value());
  return xtc;
}

void printBytes(char *start, size_t len, size_t maxPrint) {
  // I am printing nibble by nibble because when I did printf("%2.2X",*p) where p was a char *, 
  // I would not always get 2 bytes printed due to the signed nature of char, I could have used 
  // a unsigned char, that should have worked?  But I've implemented this

  static char nibbleToHexChar[] = {'0','1','2','3','4','5','6','7','8','9','A','B','C','D','E','F'};
  fprintf(stdout,"0x");
  bool addElipses = len > maxPrint;
  if (addElipses) {
    len = maxPrint;
  }
  while (len != 0) {
    len -= 1;
    uint8_t curByte = *start++;
    uint8_t highNibble = (curByte>>4) & 0xF;
    uint8_t lowNibble = curByte & 0xF;
    fprintf(stdout,"%c",nibbleToHexChar[highNibble]);
    fprintf(stdout,"%c",nibbleToHexChar[lowNibble]);
  }
  if (addElipses) fprintf(stdout,"...");
}

bool validPayload(const Pds::Damage &damage, enum Pds::TypeId::Type id) {
  if (damage.value() == 0) return true;
  if (id == Pds::TypeId::Id_EBeam) {
    bool userDamageBitSet = (damage.bits() & (1 << Pds::Damage::UserDefined));
    uint32_t otherDamageBits = (damage.bits() & (~(1 << Pds::Damage::UserDefined)));
    if (userDamageBitSet and not otherDamageBits) return true;
  }
  return false;
}

} // namespace psana_test
