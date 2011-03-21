#ifndef XTCINPUT_DGRAM_H
#define XTCINPUT_DGRAM_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Dgram.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/shared_ptr.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "pdsdata/xtc/Dgram.hh"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/**
 *  @brief Safer wrapper for Pds::Dgram class.
 *  
 *  This namespace defines smart pointer class for datagram objects and 
 *  the factory methods for creating pointers.
 *  
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace Dgram {

  typedef boost::shared_ptr<Pds::Dgram> ptr;
  
  /**
   *  @brief This method will be used in place of regular delete.
   */
  void destroy(const Pds::Dgram* dg) ;

  /**
   *  @brief Factory method which wraps existing object into a smart pointer.
   */
  ptr make_ptr(Pds::Dgram* dg) ;

  /**
   *  @brief Factory method which copies existing datagram and wraps new 
   *  object into a smart pointer.
   */
  ptr copy(Pds::Dgram* dg) ;

}

} // namespace XtcInput

#endif // XTCINPUT_DGRAM_H
