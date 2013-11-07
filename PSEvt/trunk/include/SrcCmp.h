#ifndef PSEVT_SRCCMP_H
#define PSEVT_SRCCMP_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id: 
//
// Description:
//	Class TypeInfoUtils
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Src.hh"

namespace PSEvt {
  namespace SrcCmp {
    // compare two Src objects
    int cmp(const Pds::Src& lhs, const Pds::Src& rhs);
  }
}

#endif // PSEVT_SRCCMP_H
