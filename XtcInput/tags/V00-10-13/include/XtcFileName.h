#ifndef XTCINPUT_XTCFILENAME_H
#define XTCINPUT_XTCFILENAME_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcFileName.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <iosfwd>

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

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief Representation of the XTC file name.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class XtcFileName  {
public:

  // Default constructor
  XtcFileName() ;
  
  // Construct from a full path name
  explicit XtcFileName( const std::string& path ) ;

  // Construct from dir name, experiment id, run number, stream, chunk, and if small
  XtcFileName(const std::string& dir, unsigned expNum, unsigned run, unsigned stream, unsigned chunk, bool small) ;

  // Destructor
  ~XtcFileName () {}

  // returns true if file name is not defined
  bool empty() const { return m_path.empty(); }

  // returns true if small data file
  bool small() const { return m_small; }

  // get full name
  const std::string& path() const { return m_path ; }

  // get base name
  std::string basename() const ;

  // get base name for large xtc version of file (will be the same if not small)
  std::string largeBasename() const ;

  // get base name for smalldata version of the xtc file (will be the same if small)
  std::string smallBasename() const ;

  // get file extension, anything that appears after last '.' in file name including '.' itself
  std::string extension() const ;

  // get experiment number
  unsigned expNum() const { return m_expNum ; }

  // get run number
  unsigned run() const { return m_run ; }

  // get stream number
  unsigned stream() const { return m_stream ; }

  // get chunk number
  unsigned chunk() const { return m_chunk ; }

  // compare two names
  bool operator<( const XtcFileName& other ) const ;

protected:

private:

  // Data members
  std::string m_path ;
  unsigned m_expNum ;
  unsigned m_run ;
  unsigned m_stream ;
  unsigned m_chunk ;
  bool m_small ;
};

std::ostream&
operator<<(std::ostream& out, const XtcFileName& fn);

} // namespace XtcInput

#endif // XTCINPUT_XTCFILENAME_H
