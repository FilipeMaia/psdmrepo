#ifndef PSANA_TEST_PRINTXTC_H
#define PSANA_TEST_PRINTXTC_H

#include <stdint.h>

namespace Pds {
  class Dgram;
  class Xtc;
}

namespace psana_test {
// writes dgram header to stdout.  Returns pointer to the 
// xtc object in the dgram.
Pds::Xtc * printDgramHeader(Pds::Dgram *dgram);
// translates a few enums
Pds::Xtc * printTranslatedDgramHeader(Pds::Dgram *dgram);


// writes xtc header info to stdout.  Returns xtc payload size.
uint32_t printXtcHeader(Pds::Xtc *xtc);

uint32_t printXtcWithOffsetAndDepth(Pds::Xtc *xtc,int offset, int depth);

void printBytes(char *start, size_t len, size_t maxPrint);

 void parseData(char *start, size_t len, size_t maxPrint, Pds::Xtc *xtc);

}

#endif
