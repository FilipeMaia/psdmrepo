#ifndef XTCINPUT_XTCSRCSTACK_H
#define XTCINPUT_XTCSRCSTACK_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcSrcStack.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/xtc/Src.hh"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/**
 *  Representation of the nested XTC source info.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class XtcSrcStack  {
public:

  // Default constructor
  XtcSrcStack () { m_src.reserve(5) ; }

  // Destructor
  ~XtcSrcStack () {}

  // add one level
  void push ( const Pds::Src& src ) { m_src.push_back( src ); }

  // remove top level
  void pop () { m_src.pop_back(); }

  // get topmost level
  const Pds::Src& top () const { return m_src.back() ; }

  // get the name of the source
  std::string name() const ;

protected:

private:

  // Data members
  std::vector<Pds::Src> m_src ;

};

} // namespace XtcInput

#endif // XTCINPUT_XTCSRCSTACK_H
