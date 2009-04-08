#ifndef O2OTRANSLATOR_O2OFILENAMEFACTORY_H
#define O2OTRANSLATOR_O2OFILENAMEFACTORY_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OFileNameFactory.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <map>

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

/**
 *  This class generates file names based on some template. Template
 *  string consists of the series of regular characters and any number
 *  of replaced tokens in the form '{key}'. The tokens will be replaced
 *  with the corresponding values (if any) defined by the user via
 *  addKeyword() member. Special keys are predefined:
 *    seq, seq2, seq3, seq4, seq5, seq6, seq7, seq8, seq9, seq10
 *  and these will be replaced with zero-padded value of corresponding
 *  width (1 to 10) of the seq argument given to makePath() method.
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

namespace O2OTranslator {

class O2OFileNameFactory  {
public:

  // Constructor
  O2OFileNameFactory ( const std::string& fileNameTemplate ) ;

  // Destructor
  virtual ~O2OFileNameFactory () ;

  /// add substitution keyword and its value
  virtual void addKeyword ( const std::string& key, const std::string& value ) ;

  /// generate full path name
  virtual std::string makePath ( unsigned int seq ) const ;

  /// generate hdf5-family path name, with <seq> replaced with %d
  virtual std::string makeH5Path () const ;

protected:

private:

  typedef std::map<std::string,std::string> Key2Value ;

  // Data members
  const std::string m_fileNameTemplate ;
  Key2Value m_key2value ;

  // Copy constructor and assignment are disabled by default
  O2OFileNameFactory ( const O2OFileNameFactory& ) ;
  O2OFileNameFactory& operator = ( const O2OFileNameFactory& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OFILENAMEFACTORY_H
