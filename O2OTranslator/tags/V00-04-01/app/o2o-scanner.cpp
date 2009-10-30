/**
 * Original code for XTC scanning from Chris
 *
 * $Id$
 */
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <list>

#include "pdsdata/xtc/DetInfo.hh"
#include "pdsdata/xtc/ProcInfo.hh"
#include "pdsdata/xtc/XtcIterator.hh"
#include "pdsdata/xtc/XtcFileIterator.hh"
#include "O2OTranslator/O2OXtcFileName.h"
#include "O2OTranslator/O2OXtcMerger.h"
#include "MsgLogger/MsgLogger.h"

class myLevelIter : public XtcIterator {
public:
  enum {Stop, Continue};
  myLevelIter(Xtc* xtc) : XtcIterator(xtc), _depth(1) {}
  int process(Xtc* xtc) {
    for ( unsigned i=0 ; i < _depth ; ++ i ) printf("  ");
    Level::Type level = xtc->src.level();
    printf("%s level: ",Level::name(level));
    if (level==Level::Source or level==Pds::Level::Reporter or level==Pds::Level::Control) {
      DetInfo& info = *(DetInfo*)(&xtc->src);
      printf("%s.%d %s.%d",
             DetInfo::name(info.detector()),info.detId(),
             DetInfo::name(info.device()),info.devId());
    } else {
      ProcInfo& info = *(ProcInfo*)(&xtc->src);
      printf("IpAddress %#x ProcessId %#x",info.ipAddr(),info.processId());
    }
    if (xtc->contains.id() == TypeId::Id_Xtc ) {
      printf("\n");
      ++_depth;
      this->iterate( xtc );
      --_depth;
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
  fprintf(stderr,"Usage: %s [-h] <filename> ...\n", progname);
}

int main(int argc, char* argv[]) {
  int c;
  int parseErr = 0;
  int verbose = 0 ;

  while ((c = getopt(argc, argv, "hv")) != -1) {
    switch (c) {
    case 'h':
      usage(argv[0]);
      exit(0);
    case 'v':
      ++ verbose ;
      break ;
    default:
      parseErr++;
    }
  }

  MsgLogger::MsgLogLevel loglvl ( 3 - verbose ) ;
  MsgLogger::MsgLogger rootlogger ;
  rootlogger.setLevel ( loglvl ) ;

  std::list<O2OTranslator::O2OXtcFileName> files ;
  for ( int i = optind ; i < argc ; ++ i ) {
    files.push_back( O2OTranslator::O2OXtcFileName(argv[i]) ) ;
  }

  if (files.empty()) {
    usage(argv[0]);
    exit(2);
  }

  O2OTranslator::O2OXtcMerger iter(files,0x100000,O2OTranslator::O2OXtcMerger::OneStream);
  while ( Dgram* dg = iter.next() ) {
    const Pds::Sequence& seq = dg->seq ;
    const Pds::ClockTime& clock = seq.clock() ;
    const Pds::TimeStamp& stamp = seq.stamp() ;
    printf("%s transition: type %d, time %u sec %u nsec, ticks %u, fiducials %u, control %u, payloadSize %d\n",
           TransitionId::name(seq.service()),
           int(seq.type()),
           clock.seconds(), clock.nanoseconds(),
           stamp.ticks(),stamp.fiducials(),stamp.control(),
           dg->xtc.sizeofPayload());
    myLevelIter iter(&(dg->xtc));
    iter.iterate();

    delete [] (char*)dg ;
  }

  return 0;
}
