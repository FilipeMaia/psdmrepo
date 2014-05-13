#include <stdio.h>  
#include <unistd.h>  // open, close
#include <fcntl.h>   // O_RDONLY
#include <stdlib.h>  // atoi
#include <stddef.h>  // offsetof
#include <string.h>  // strcmp

#include <string>
#include <map>

#include "psana_test/xtciter.h"
#include "psana_test/printxtc.h"

#include "pdsdata/xtc/Dgram.hh"


const char * usage = "%s dg|xtc1|xtc xtcfile [--payload=n] [--dgrams=n]\n \
  dumps xtc files based on first argument:\n\
      dg:   dgram headers (with offset in file)\n\
      xtc1: dgram headers and xtc in dgram (not xtc children)\n\
      xtc:  all xtc, dump payload in hex.\n\
      \n\
    arg2 is the xtcfile to dump. \n\
    The optional last arguments:\n\
      --payload=n    how many bytes of the payload to print for xtc\n\
      --dgrams=n     how many dgrams to print\n ";

void dgramHeaderIterator_nextAndOffset(int fd, long maxDgrams);           // dg
void dgramAndItsXtcIterator(int fd, long maxDgrams);                      // xtc1
void dgramAndXtcChildrenIteratorPrintXtcOffsetAndBytes(int fd, long maxDgrams, size_t maxPayloadPrint); // xtc

const int DEFAULT_MAX_PAYLOAD_PRINT = 20;

int main(int argc, char *argv[]) {
  char * dumpArg = NULL;
  std::string xtcFileName;
  size_t maxPayloadPrint = DEFAULT_MAX_PAYLOAD_PRINT;
  long maxDgrams = -1;

  if (argc > 1) dumpArg = argv[1];
  if (argc > 2)  xtcFileName = argv[2];
  if ((NULL == dumpArg) or (xtcFileName=="")) {
    printf(usage,argv[0]);
    return -1;
  }
  // check for --payload=n and --dgrams=n
  static const char *payload = "--payload=";
  static const char *dgrams = "--dgrams=";
  for (int ii = 3; ii < argc; ii++) {
    bool isPayload = (0 == strncmp(argv[ii], payload, strlen(payload)));
    bool isDgrams = (0 == strncmp(argv[ii], dgrams, strlen(dgrams)));
    if (not isPayload and not isDgrams) {
      fprintf(stderr,"Error: argument %s starts out with neither %s nor %s\n",
              argv[ii], payload, dgrams);
      return -1;
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
    dgramAndXtcChildrenIteratorPrintXtcOffsetAndBytes(fd, maxDgrams, maxPayloadPrint);
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

void dgramAndXtcChildrenIteratorPrintXtcOffsetAndBytes(int fd, long maxDgrams, size_t maxPayloadPrint) {
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
        fprintf(stdout," plen=%d payload=",xtc->sizeofPayload());
        if (xtc->damage.value()!=0) {
          fprintf(stdout,"**damaged**");
        } else {
          psana_test::printBytes(xtc->payload(), xtc->sizeofPayload(), maxPayloadPrint);
        }
      }
      fprintf(stdout,"\n");
      xtcDepthOffset = xtcIter.nextWithPos();
    }
    if ((maxDgrams > 0) and (dgNumber >= maxDgrams)) return;
    dgramOffset = dgIter.nextAndOffsetFromStart();
  }
}


