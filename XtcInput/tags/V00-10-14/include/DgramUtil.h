#ifndef XTCINPUT_DGRAMUTIL_H
#define XTCINPUT_DGRAMUTIL_H

#include <vector>
#include "XtcInput/Dgram.h"

namespace XtcInput {

/// returns true if all datagrams have the same transition
bool allDgsHaveSameTransition(const std::vector<XtcInput::Dgram> &dgs);

/**
 * @brief returns true if the list of datagrams pass l3accept.
 *
 * Expects a list of datagrams, where some are from DAQ streams, and some from
 * Control streams. Checks the DAQ streams for trimmed datagrams. If all are trimmed,
 * returns false. If one is not trimmed, returns true. Control streams are ignored
 * in this case.
 *
 * If there are no datagrams from DAQ streams, only control streams, than it is an
 * automatic pass and true is returned. Control streams do not specify if trimmed.
 */
bool l3tAcceptPass(const std::vector<XtcInput::Dgram>& dgs, int firstControlStream);

// orders Dgram's by stream number.
class LessStream {
 public:
  LessStream() {}
  bool operator()(const XtcInput::Dgram &a, const XtcInput::Dgram &b) {
    return a.file().stream() < b.file().stream();
  }
};
 
}; // namespace XtcInput

#endif

