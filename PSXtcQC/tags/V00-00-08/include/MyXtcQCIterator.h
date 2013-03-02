#ifndef PSXTCQC_MYXTCQCITERATOR_H
#define PSXTCQC_MYXTCQCITERATOR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class MyXtcQCIterator.
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

#include "PSXtcQC/XtcQCIterator.h"
#include "PSXtcQC/QCStatistics.h"

#include "pdsdata/xtc/Xtc.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace PSXtcQC {

/// @addtogroup PSXtcQC

/**
 *  @ingroup PSXtcQC
 *
 *  @brief C++ source file code template.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 */

class MyXtcQCIterator : public PSXtcQC::XtcQCIterator {
public:
  enum {Stop, Continue};
  MyXtcQCIterator(Pds::Xtc* xtc, PSXtcQC::QCStatistics* qcstat, unsigned ndgram, unsigned depth=0) : PSXtcQC::XtcQCIterator(xtc), _depth(depth), _ndgram(ndgram), _qcstat(qcstat) {}

  //-----------------

  void iterateNextLevel() { return iterateNextLevel(root()); }

  //-----------------

  void iterateNextLevel(Pds::Xtc* xtc) {
      MyXtcQCIterator iter(xtc, _qcstat, _ndgram, _depth+1);
      iter.iterate();
  }

  //-----------------
  // Implementation for virtual int XtcQCIterator::process(Pds::Xtc* xtc);
  int process(Pds::Xtc* xtc) 
  {
    return _qcstat->processXTC(xtc, _depth, _ndgram);
  }

  //-----------------

  void processSizeError(Pds::Xtc* root, Pds::Xtc* xtc, int remaining)
  {
    return _qcstat->processXTCSizeError(root, xtc, remaining, _depth, _ndgram);
  }

  //-----------------

private:
  unsigned _depth;
  unsigned _ndgram;
  PSXtcQC::QCStatistics* _qcstat; 
};


} // namespace PSXtcQC

#endif // PSXTCQC_MYXTCQCITERATOR_H
