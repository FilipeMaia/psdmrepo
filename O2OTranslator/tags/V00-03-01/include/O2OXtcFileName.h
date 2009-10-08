#ifndef O2OTRANSLATOR_O2OXTCFILENAME_H
#define O2OTRANSLATOR_O2OXTCFILENAME_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OXtcFileName.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>

//----------------------
// Base Class Headers --
//----------------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Representation of the XTC file name.
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

class O2OXtcFileName  {
public:

  // Default constructor
  O2OXtcFileName() ;
  O2OXtcFileName( const std::string& path ) ;

  // Destructor
  ~O2OXtcFileName () {}

  // get full name
  const std::string& path() const { return m_path ; }

  // get base name
  std::string basename() const ;

  // get experiment number
  unsigned expNum() const { return m_expNum ; }

  // get run number
  unsigned run() const { return m_run ; }

  // get stream number
  unsigned stream() const { return m_stream ; }

  // get chunk number
  unsigned chunk() const { return m_chunk ; }

  // compare two names
  bool operator<( const O2OXtcFileName& other ) const ;

protected:

  // helper methods
  unsigned _cvt ( const char* ptr, bool& stat ) const ;

private:

  // Data members
  std::string m_path ;
  unsigned m_expNum ;
  unsigned m_run ;
  unsigned m_stream ;
  unsigned m_chunk ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OXTCFILENAME_H
