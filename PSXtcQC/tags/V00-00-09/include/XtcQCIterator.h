#ifndef PSXTCQC_XTCQCITERATOR_H
#define PSXTCQC_XTCQCITERATOR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcQCIterator.
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


class XtcQCIterator  {
public:
    XtcQCIterator(Pds::Xtc* root);
    XtcQCIterator() {}
    virtual ~XtcQCIterator() {}

    virtual int  process(Pds::Xtc* xtc) = 0;
    virtual void processSizeError(Pds::Xtc* root, Pds::Xtc* xtc, int remaining) = 0;
    virtual void iterateNextLevel(Pds::Xtc* xtc) = 0;
    virtual void iterateNextLevel() = 0;

    void iterate();
    void iterate(Pds::Xtc*);
    Pds::Xtc* root() const {return _root;} 

private:
    Pds::Xtc* _root; // Collection to process in the absence of an argument...
 
    // Copy constructor and assignment are disabled by default
    XtcQCIterator ( const XtcQCIterator& );
    XtcQCIterator& operator = ( const XtcQCIterator& );
};


} // namespace PSXtcQC

using namespace PSXtcQC;

inline XtcQCIterator::XtcQCIterator(Pds::Xtc* root) : _root(root) {}
inline void XtcQCIterator::iterate() {iterate(_root);} 

#endif // PSXTCQC_XTCQCITERATOR_H
