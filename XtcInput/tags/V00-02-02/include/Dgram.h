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

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Dgram.hh"
#include "XtcInput/XtcFileName.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/**
 *  @brief Wrapper for Pds::Dgram class.
 *  
 *  This class wraps Pds::Datagram class and also adds some additional 
 *  context information to it such as file name and position.
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

class Dgram {
public:
  
  typedef boost::shared_ptr<Pds::Dgram> ptr;

  /**
   *  @brief Factory method which wraps existing object into a smart pointer.
   */
  static ptr make_ptr(Pds::Dgram* dg) ;

  /**
   *  Constructor takes a smart pointer to XTC datagram object and
   *  the file name where datagram has originated.
   */
  Dgram(const ptr& dg, XtcFileName file) : m_dg(dg), m_file(file) {}

  /**
   *  Default ctor
   */
  Dgram() : m_dg(), m_file() {}

  /// Return pointer to the datagream
  ptr dg() const { return m_dg; }
  
  /// Return file name
  const XtcFileName& file() const { return m_file; }

  bool empty() const { return not m_dg.get(); }
  
private:
  
  /**
   *  @brief This method will be used in place of regular delete.
   */
  static void destroy(const Pds::Dgram* dg) ;

  /**
   *  @brief Factory method which copies existing datagram and wraps new 
   *  object into a smart pointer.
   */
  static ptr copy(Pds::Dgram* dg) ;

  // Data members
  ptr m_dg;
  XtcFileName m_file;
};

} // namespace XtcInput

#endif // XTCINPUT_DGRAM_H
