#ifndef O2OTRANSLATOR_O2OXTCITERATOR_H
#define O2OTRANSLATOR_O2OXTCITERATOR_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcIterator.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <map>

//----------------------
// Base Class Headers --
//----------------------
#include "pdsdata/xtc/XtcIterator.hh"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "O2OTranslator/O2OXtcSrc.h"
#include "pdsdata/xtc/TypeId.hh"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  Subclass of XtcIterator which forwards all
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class O2OXtcScannerI ;
class O2ODataTypeCvtI ;

class O2OXtcIterator : public Pds::XtcIterator {
public:

  // Default constructor
  O2OXtcIterator ( Xtc* xtc, O2OXtcScannerI* scanner, bool config = false ) ;

  // Destructor
  virtual ~O2OXtcIterator () ;

  // process one sub-XTC, returns >0 for success, 0 for error
  virtual int process(Xtc* xtc) ;

protected:

  // process one sub-XTC, returns >0 for success, 0 for error
  int process_int(Xtc* xtc) ;

private:

  // Data members
  O2OXtcScannerI* m_scanner ;
  O2OXtcSrc m_src ;
  bool m_config;
  std::map<Pds::Src, std::vector<Pds::TypeId> > m_typeIds;   // Stores list of types stored per device

  // Copy constructor and assignment are disabled by default
  O2OXtcIterator ( const O2OXtcIterator& ) ;
  O2OXtcIterator operator = ( const O2OXtcIterator& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OXTCITERATOR_H
