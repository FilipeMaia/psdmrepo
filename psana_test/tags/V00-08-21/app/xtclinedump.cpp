#include <stdio.h>  
#include <unistd.h>  // open, close
#include <fcntl.h>   // O_RDONLY
#include <stdlib.h>  // atoi
#include <stddef.h>  // offsetof
#include <string.h>  // strcmp

#include <string>
#include <map>

#include "pdsdata/xtc/Dgram.hh"
#include "pdsdata/xtc/TypeId.hh"
#include "pdsdata/psddl/epics.ddl.h"
#include "psana_test/xtciter.h"
#include "psana_test/printxtc.h"
#include "psana_test/epics2str.h"
#include "psana_test/smldata.h"

const char * usage = "%s dg|xtc1|xtc xtcfile [--payload=n] [--dgrams=n]\n \
  dumps xtc files based on first argument:\n\
      dg:   dgram headers (with offset in file)\n\
      xtc1: dgram headers and xtc in dgram (not xtc children)\n\
      xtc:  all xtc, dump payload in hex.\n\
      \n\
    arg2 is the xtcfile to dump. \n\
    The optional last arguments:\n\
      --payload=n    how many bytes of the payload to print for xtc\n\
      --dgrams=n     how many dgrams to print\n\
      --sml          parse out the small data proxies instead of printing the payload\n\
      --epics        print extra lines with details on epics, both epicsConfigV1 and the pv's\n";

void dgramHeaderIterator_nextAndOffset(int fd, long maxDgrams);           // dg
void dgramAndItsXtcIterator(int fd, long maxDgrams);                      // xtc1
void dgramAndXtcChildrenIteratorPrintXtcOffsetAndBytes(int fd, long maxDgrams, size_t maxPayloadPrint,
                                                       bool printEpicsConfig, bool parseSml); // xtc

const int DEFAULT_MAX_PAYLOAD_PRINT = 1;

int main(int argc, char *argv[]) {
  char * dumpArg = NULL;
  std::string xtcFileName;
  size_t maxPayloadPrint = DEFAULT_MAX_PAYLOAD_PRINT;
  long maxDgrams = -1;
  bool printEpics = false;
  bool parseSml = false;
  if (argc > 1) dumpArg = argv[1];
  if (argc > 2)  xtcFileName = argv[2];
  if ((NULL == dumpArg) or (xtcFileName=="")) {
    printf(usage,argv[0]);
    return -1;
  }
  // check for --payload=n , --dgrams=n, --epics and --sml
  static const char *payload = "--payload=";
  static const char *dgrams = "--dgrams=";
  static const char *epics = "--epics";
  static const char *sml = "--sml";
  for (int ii = 3; ii < argc; ii++) {
    bool isPayload = (0 == strncmp(argv[ii], payload, strlen(payload)));
    bool isDgrams = (0 == strncmp(argv[ii], dgrams, strlen(dgrams)));
    bool isEpics = (0 == strncmp(argv[ii], epics, strlen(epics)));
    bool isSml = (0 == strncmp(argv[ii], sml, strlen(sml)));
    if (not isPayload and not isDgrams and not isEpics and not isSml) {
      fprintf(stderr,"Error: argument %s starts out with neither %s , %s,  %s nor %s\n",
              argv[ii], payload, dgrams, epics, sml);
      return -1;
    }
    if (isEpics) {
      printEpics = true;
      continue;
    }
    if (isSml) {
      parseSml = true;
      continue;
    }
    char * intVal = argv[ii];
    if (isPayload) intVal += strlen(payload);
    if (isDgrams) intVal += strlen(dgrams);
    char *p = intVal;
    // check for digits
    do {
      if ((*p < '0') or (*p > '9')) {
        fprintf(stderr,"arg %s does not contain an integer after the = sign\n", argv[ii]);
        return -1;
      }
      p++;
    } while (*p != 0);
    long val = atol(intVal);
    if (isDgrams) maxDgrams = val;
    if (isPayload) maxPayloadPrint = val;
  }
  int fd = ::open(xtcFileName.c_str(), O_RDONLY);
  if (fd == -1) {
    fprintf(stderr,"Error opening xtc file %s\n",xtcFileName.c_str());
    return -1;
  }

  if (strcmp(dumpArg,"dg")==0) {
    dgramHeaderIterator_nextAndOffset(fd, maxDgrams);
  } else if (strcmp(dumpArg,"xtc1")==0) {
    dgramAndItsXtcIterator(fd, maxDgrams);
  } else if (strcmp(dumpArg,"xtc")==0) {
    dgramAndXtcChildrenIteratorPrintXtcOffsetAndBytes(fd, maxDgrams, maxPayloadPrint, printEpics, parseSml);
  } else {
    fprintf(stderr, "ERROR: unexpected, dumpArg=%s not recognized, must be one of 'dg', 'xtc1', 'xtc', 'xtcp'\n",dumpArg);
    fprintf(stderr,usage,argv[0]);
    ::close(fd);
    return -1;
  }
  ::close(fd);
  return 0;
}

void dgramHeaderIterator_nextAndOffset(int fd, long maxDgrams) {
  psana_test::DgramHeaderIterator dgIter(fd);
  std::pair<Pds::Dgram *,size_t> dgramOffset = dgIter.nextAndOffsetFromStart();
  int dgNumber = 0;
  while (dgramOffset.first) {
    dgNumber ++;
    Pds::Dgram * dgram = dgramOffset.first;
    size_t offset = dgramOffset.second;
    fprintf(stdout,"dg=%5d offset=0x%8.8lX ",dgNumber,offset);
    psana_test::printTranslatedDgramHeader(dgram);
    fprintf(stdout,"\n");
    if ((maxDgrams > 0) and (dgNumber >= maxDgrams)) return;
    dgramOffset = dgIter.nextAndOffsetFromStart();
  }
}

void dgramAndItsXtcIterator(int fd, long maxDgrams) {
  psana_test::DgramWithXtcPayloadIterator dgIter(fd);
  std::pair<Pds::Dgram *,size_t> dgramOffset = dgIter.nextAndOffsetFromStart();
  int dgNumber = 0;
  while (dgramOffset.first) {
    dgNumber ++;
    Pds::Dgram * dgram = dgramOffset.first;
    size_t offset = dgramOffset.second;
    fprintf(stdout,"dg=%5d offset=0x%8.8lX ",dgNumber,offset);
    psana_test::printTranslatedDgramHeader(dgram);
    fprintf(stdout,"\n  xtc: ");
    psana_test::printXtcHeader(& dgram->xtc);
    fprintf(stdout,"\n");
    if ((maxDgrams > 0) and (dgNumber >= maxDgrams)) return;
    dgramOffset = dgIter.nextAndOffsetFromStart();
  }
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

void dgramAndXtcChildrenIteratorPrintXtcOffsetAndBytes(int fd, long maxDgrams, size_t maxPayloadPrint, 
                                                       bool printEpics, bool parseSml) {
  psana_test::DgramWithXtcPayloadIterator dgIter(fd);
  std::pair<Pds::Dgram *,size_t> dgramOffset = dgIter.nextAndOffsetFromStart();
  int dgNumber = 0;
  while (dgramOffset.first) {
    dgNumber ++;
    Pds::Dgram * dgram = dgramOffset.first;
    size_t dgOffset = dgramOffset.second;
    fprintf(stdout,"dg=%5d offset=0x%8.8lX ",dgNumber,dgOffset);
    psana_test::printTranslatedDgramHeader(dgram);
    size_t rootXtcOffset = dgOffset + size_t(((uint8_t *)(&dgram->xtc))-((uint8_t*)dgram));
    fprintf(stdout,"\nxtc d=0  offset=0x%8.8lX ",rootXtcOffset);
    psana_test::printXtcHeader(& dgram->xtc);
    fprintf(stdout,"\n");
    psana_test::XtcChildrenIterator xtcIter(&dgram->xtc,1);  // starting depth of 1 for children
    psana_test::XtcDepthOffset xtcDepthOffset = xtcIter.nextWithPos();
    while (xtcDepthOffset.xtc) {
      fprintf(stdout,"xtc d=%d  offset=0x%8.8lX ",xtcDepthOffset.depth,rootXtcOffset + xtcDepthOffset.offset);
      psana_test::printXtcHeader(xtcDepthOffset.xtc);
      Pds::Xtc *xtc = xtcDepthOffset.xtc;
      if (xtc->contains.id() != Pds::TypeId::Id_Xtc) {
        fprintf(stdout," plen=%d",xtc->sizeofPayload());
        const Pds::TypeId &typeId = xtc->contains;
        const Pds::Damage damage = xtc->damage;
        if (validPayload(damage,typeId.id())) {
          if ((not printEpics) and (typeId.id() == Pds::TypeId::Id_Epics)) {
            // if printEpics is true, a more detailed line will follow this one
            //   and we do not have to do this
            Pds::Epics::EpicsPvHeader *pv = 
              static_cast<Pds::Epics::EpicsPvHeader *>(static_cast<void *>(xtc->payload()));
            fprintf(stdout," dbr=%d", pv->dbrType());
            fprintf(stdout," numElem=%d", pv->numElements());
            fprintf(stdout," pvId=%d", pv->pvId());
            if (pv->isCtrl()) {
              Pds::Epics::EpicsPvCtrlHeader *ctrlPv = 
                static_cast<Pds::Epics::EpicsPvCtrlHeader *>(static_cast<void *>(xtc->payload()));
              fprintf(stdout," pvName=%s", ctrlPv->pvName());
            }
          }
          if (parseSml and isSmallData(xtc->contains)) {
            parseSmallData(stdout, xtc->contains, xtc->payload());
          } else {
            fprintf(stdout," payload=");
            psana_test::printBytes(xtc->payload(), xtc->sizeofPayload(), maxPayloadPrint);
          }
        } else {
          fprintf(stdout," payload=**damaged**");
        }
      }
      fprintf(stdout,"\n");
      if (printEpics) {
        const Pds::TypeId &typeId = xtc->contains;
        if (typeId.id() == Pds::TypeId::Id_EpicsConfig) {
          if (typeId.version() == 1 ) {
            const Pds::Damage damage = xtc->damage;
            if (validPayload(damage,typeId.id())) {
              Pds::Epics::ConfigV1 *cfg = static_cast<Pds::Epics::ConfigV1 *>(static_cast<void *>(xtc->payload()));
              ndarray<const Pds::Epics::PvConfigV1,1> pvCfg = cfg->getPvConfig();
              for (int32_t idx = 0; idx < cfg->numPv(); ++idx) {
                fprintf(stdout,"  pvId=%3d  descr=%s\n", int(pvCfg[idx].pvId()), pvCfg[idx].description());
              }
            }
          }
        } else if (typeId.id() == Pds::TypeId::Id_Epics) {
          std::string epicsString = psana_test::epics2Str(xtc);
          const char *c_str = epicsString.c_str();
          fprintf(stdout, "  epics: %s\n",c_str);
        }
      }
      xtcDepthOffset = xtcIter.nextWithPos();
    }
    if ((maxDgrams > 0) and (dgNumber >= maxDgrams)) return;
    dgramOffset = dgIter.nextAndOffsetFromStart();
  }
}


