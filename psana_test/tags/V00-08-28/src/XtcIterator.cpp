// copied from pdsdata

/*
 ** ++
 **  Package:
 **	OdfContainer
 **
 **  Abstract:
 **     non-inline functions for "InXtcIterator.h"
 **
 **  Author:
 **      Michael Huffer, SLAC, (415) 926-4269
 **
 **  Creation Date:
 **	000 - October 11,1998
 **
 **  Revision History:
 **	None.
 **
 ** --
 */

#include <stdio.h>

#include "psana_test/XtcIterator.h"
#include "pdsdata/xtc/Xtc.hh"

using namespace Pds;

using namespace psana_test;

/*
 ** ++
 **
 **   Iterate over the collection specifed as an argument to the function.
 **   For each "Xtc" found call back the "process" function. If the
 **   "process" function returns zero (0) the iteration is aborted and
 **   control is returned to the caller. Otherwise, control is returned
 **   when all elements of the collection have been scanned.
 **
 ** --
 */

void XtcIterator::iterate(Xtc* root) 
{
  if (root->damage.value() & ( 1 << Damage::IncompleteContribution)) {
    if (_diagnose) {
      fprintf(stderr, "xtc object has Damage::IncompleteContribution.\n");
    }
    return;
  }

  Xtc* xtc     = (Xtc*)root->payload();
  int remaining = root->sizeofPayload();

  while(remaining > 0)
  {
    if(xtc->extent==0) {
      if (_diagnose) {
        fprintf(stderr, "*There is a xtc object with a 0 extent.*\n");
      }
      break; // try to skip corrupt event
    }
    if(xtc->extent > _maxXtcExtent) {
      fprintf(stderr, "ERROR: There is an xtc object with an extent > %u\n", _maxXtcExtent);
      break;
    }
    if(!process(xtc)) break;
    remaining -= xtc->sizeofPayload() + sizeof(Xtc);
    xtc      = xtc->next();
  }

  return;
}
