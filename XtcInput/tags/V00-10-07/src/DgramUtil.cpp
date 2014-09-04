//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	utility functions InputModules that work with Dgram's
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/DgramUtil.h"

//-----------------
// C/C++ Headers --
//-----------------
#include "boost/foreach.hpp"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/L1AcceptEnv.hh"

bool XtcInput::allDgsHaveSameTransition(const std::vector<XtcInput::Dgram> &dgs) {
  Pds::TransitionId::Value last = Pds::TransitionId::Unknown;
  BOOST_FOREACH(const XtcInput::Dgram& dg, dgs) {
    if (not dg.empty()) {
      if ((last != Pds::TransitionId::Unknown) and (last != dg.dg()->seq.service())) {
        return false;
      }
      last = dg.dg()->seq.service();
    }
  }
  return true;
}

bool XtcInput::l3tAcceptPass(const std::vector<XtcInput::Dgram>& dgs, int firstControlStream)
{
  bool foundDaq = false;
  // if at least one DAQ stream is not trimmed, or there are no DAQ streams, then we accept all
  BOOST_FOREACH(const XtcInput::Dgram& dg, dgs) {
    if ( int(dg.file().stream()) < firstControlStream ) {
      foundDaq = true;
      if (not static_cast<const Pds::L1AcceptEnv&>(dg.dg()->env).trimmed()) return true;
    }
  }
  if (foundDaq) return false;  // all DAQ are trimmed
  return true;
}
