#ifndef TRANSLATOR_LESSEVENTIDPTRS_H
#define TRANSLATOR_LESSEVENTIDPTRS_H

#include "boost/shared_ptr.hpp"
#include "PSEvt/EventId.h"

class LessEventIdPtrs {
public:
  bool operator()(const boost::shared_ptr<PSEvt::EventId> & a,
                  const boost::shared_ptr<PSEvt::EventId> & b) {
    return *a < *b;
  }
};

#endif
