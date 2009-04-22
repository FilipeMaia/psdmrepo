/**
 * Original code for XTC scanning from Chris
 *
 * $Id$
 */
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/ProcInfo.hh"
#include "pdsdata/xtc/XtcIterator.hh"
#include "pdsdata/xtc/XtcFileIterator.hh"

class myLevelIter : public XtcIterator {
public:
  enum {Stop, Continue};
  myLevelIter(Xtc* xtc, unsigned depth) : XtcIterator(xtc), _depth(depth) {}
  int process(Xtc* xtc) {
    unsigned i=_depth; while (i--) printf("  ");
    Level::Type level = xtc->src.level();
    printf("%s level: ",Level::name(level));
    if (level==Level::Source) {
      DetInfo& info = *(DetInfo*)(&xtc->src);
      printf("%s%d %s%d",
             DetInfo::name(info.detector()),info.detId(),
             DetInfo::name(info.device()),info.devId());
    } else {
      ProcInfo& info = *(ProcInfo*)(&xtc->src);
      printf("IpAddress %#x ProcessId %#x",info.ipAddr(),info.processId());
    }
    if (xtc->contains.id() == TypeId::Id_Xtc ) {
      printf("\n");
      myLevelIter iter(xtc,_depth+1);
      iter.iterate();
    } else {
      printf(" id=%d name=%s version=%d\n", xtc->contains.id(),
          Pds::TypeId::name(xtc->contains.id()), xtc->contains.version() );
    }
    return Continue;
  }
private:
  unsigned _depth;
};

void usage(char* progname) {
  fprintf(stderr,"Usage: %s -f <filename> [-h]\n", progname);
}

int main(int argc, char* argv[]) {
  int c;
  char* xtcname=0;
  int parseErr = 0;

  while ((c = getopt(argc, argv, "hf:")) != -1) {
    switch (c) {
    case 'h':
      usage(argv[0]);
      exit(0);
    case 'f':
      xtcname = optarg;
      break;
    default:
      parseErr++;
    }
  }

  if (!xtcname) {
    usage(argv[0]);
    exit(2);
  }

  FILE* file = fopen(xtcname,"r");
  if (!file) {
    perror("Unable to open file %s\n");
    exit(2);
  }

  XtcFileIterator iter(file,0x1000000);
  while ( Dgram* dg = iter.next() ) {
    const Pds::Sequence& seq = dg->seq ;
    const Pds::ClockTime& clock = seq.clock() ;
    printf("%s transition: type %d, time %u sec %u nsec, hi/lo 0x%x/0x%x, payloadSize %d\n",
           TransitionId::name(seq.service()),
           int(seq.type()),
           clock.seconds(), clock.nanoseconds(),
           seq.high(),seq.low(),
           dg->xtc.sizeofPayload());
    myLevelIter iter(&(dg->xtc),1);
    iter.iterate();
  }

  fclose(file);
  return 0;
}
