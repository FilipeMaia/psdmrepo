//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcQCIterator...
//
// Author List:
//      Mikhail S. Dubrovin
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "PSXtcQC/XtcQCIterator.h"
//-----------------
// C/C++ Headers --
//-----------------
#include <iostream> // for cout, puts etc.

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

using namespace PSXtcQC;
using namespace Pds;

namespace PSXtcQC {
//===================

void XtcQCIterator::iterate(Pds::Xtc* root) 
{
    if (!process(root)) return;
    if (root->contains.id() != Pds::TypeId::Id_Xtc) return; 

    Pds::Xtc* xtc = (Pds::Xtc*)root->payload(); // get the 1st inserted in root xtc.
    int remaining = root->sizeofPayload();

    while(true) {

      if(remaining==0) break; // normal completion of the loop

      // Negative remaining catcher
      if( remaining < 0 ) {
         std::cout << "\nXtcQCIterator::iterate(...): ERROR!!! Enclosed data size exceeds available payload, remaining=" << remaining;
	 processSizeError(root, xtc, remaining);
	 return;
      }

      iterateNextLevel(xtc);
      remaining -= xtc->sizeofPayload() + sizeof(Pds::Xtc);
      xtc = xtc->next();
      //std::cout << " remaining:" << remaining << "\n";
    }

    return;
}

//===================

} // namespace PSXtcQC
