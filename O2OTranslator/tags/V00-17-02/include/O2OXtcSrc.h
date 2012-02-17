#ifndef O2OTRANSLATOR_O2OXTCSRC_H
#define O2OTRANSLATOR_O2OXTCSRC_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcSrc.
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

namespace O2OTranslator {

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

class O2OXtcSrc  {
public:

  // Default constructor
  O2OXtcSrc () { m_src.reserve(5) ; }

  // Destructor
  ~O2OXtcSrc () {}

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

  // Copy constructor and assignment are disabled by default
  O2OXtcSrc ( const O2OXtcSrc& ) ;
  O2OXtcSrc& operator = ( const O2OXtcSrc& ) ;

};

} // namespace O2OTranslator

/// Helper operator to format Pds::Src to a standard stream
std::ostream&
operator<<(std::ostream& out, const Pds::Src& src);

#endif // O2OTRANSLATOR_O2OXTCSRC_H
