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
#include <boost/utility.hpp>

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

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  This class generates file names based on some template. Template
 *  string consists of the series of regular characters and any number
 *  of replaced tokens in the form '{key}'. The tokens will be replaced
 *  with the corresponding values (if any) defined by the user via
 *  addKeyword() member. Special keys are predefined:
 *    seq, seq2, seq3, seq4, seq5, seq6, seq7, seq8, seq9, seq10;
 *  and these will be replaced with zero-padded value of corresponding
 *  width (1 to 10) of the seq argument given to makePath() method.
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class O2OFileNameFactory : boost::noncopyable {
public:

  // Constructor
  O2OFileNameFactory ( const std::string& fileNameTemplate ) ;

  // Destructor
  virtual ~O2OFileNameFactory () ;

  /// add substitution keyword and its value
  virtual void addKeyword ( const std::string& key, const std::string& value ) ;

  /**
   *   @brief Generate file path name
   *
   *   If seq is negative then {seq} replaced with %d, {seq1} with %01d,
   *   {seq2} with %02d, etc. Otherwise these will be replaced with the
   *   zero-padded value of the argument.
   */
  virtual std::string makePath(int seq) const;

protected:

private:

  typedef std::map<std::string, std::string> Key2Value ;

  // Data members
  const std::string m_fileNameTemplate ;
  Key2Value m_key2value ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OFILENAMEFACTORY_H
