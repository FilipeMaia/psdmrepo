#ifndef PSXTCINPUT_DGRAMPIECES_H
#define PSXTCINPUT_DGRAMPIECES_H

#include <vector>
#include "XtcInput/Dgram.h"

namespace PSXtcInput {

class DgramPieces {
public:
  void reset() {eventDg.clear(); nonEventDg.clear();}
  std::vector<XtcInput::Dgram> eventDg;
  std::vector<XtcInput::Dgram> nonEventDg;
};

}

#endif // PSXTCINPUT_DGRAMPIECES_H
