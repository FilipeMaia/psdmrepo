#ifndef O2OTRANSLATOR_O2OMETADATA_H
#define O2OTRANSLATOR_O2OMETADATA_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2OMetaData.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <map>
#include <list>

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
 *  This is a container class for all metadata available to translator.
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

class O2OMetaData  {
public:

  typedef std::map<std::string,std::string> cont_type ;
  typedef cont_type::const_iterator const_iterator;

  // Default constructor
  O2OMetaData ( unsigned long runNumber,
                const std::string& runType,
                const std::string& instrument,
                const std::string& experiment,
                const std::string& calibDir,
                const std::list<std::string>& extraMetaData ) ;

  // Destructor
  ~O2OMetaData () ;

  // get run number, 0 if undefined
  unsigned long runNumber() const { return m_runNumber ; }

  // get run type or empty string
  const std::string& runType() const { return m_runType ; }

  // get experiment name or empty string
  const std::string& experiment() const { return m_experiment ; }

  // get instrument name or empty string
  const std::string& instrument() const { return m_instrument ; }

  // get path for the calibration directory
  const std::string& calibDir() const { return m_calibDir ; }

  // get the iterators for extra meta data
  const_iterator extra_begin() const { return m_extraMetaData.begin() ; }
  const_iterator extra_end() const { return m_extraMetaData.end() ; }

protected:

private:

  // Data members
  unsigned long m_runNumber ;
  const std::string m_runType ;
  const std::string m_instrument ;
  const std::string m_experiment ;
  const std::string m_calibDir ;
  std::map<std::string,std::string> m_extraMetaData ;

  // Copy constructor and assignment are disabled by default
  O2OMetaData ( const O2OMetaData& ) ;
  O2OMetaData& operator = ( const O2OMetaData& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2OMETADATA_H
