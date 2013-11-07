#include "PSEvt/SrcCmp.h"

using namespace PSEvt;

int SrcCmp::cmp(const Pds::Src& lhs, const Pds::Src& rhs)
{
  // ignore PID in comparison
  int diff = int(lhs.level()) - int(rhs.level());
  if (diff != 0) return diff;
  return int(lhs.phy()) - int(rhs.phy());
}
