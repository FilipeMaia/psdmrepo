#ifndef O2OTRANSLATOR_O2ONEXUSWRITER_H
#define O2OTRANSLATOR_O2ONEXUSWRITER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class O2ONexusWriter.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <set>

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/O2OXtcScannerI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "nexuspp/NxppFile.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  Scanner class which sends all data to Nexus file
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace O2OTranslator {

class O2ONexusWriter : public O2OXtcScannerI {
public:

  // Default constructor
  O2ONexusWriter ( const std::string& fileName ) ;

  // Destructor
  virtual ~O2ONexusWriter () ;

  // signal start/end of the event
  virtual void eventStart ( const Pds::Sequence& seq ) ;
  virtual void eventEnd ( const Pds::Sequence& seq ) ;

  // signal start/end of the level
  virtual void levelStart ( const Pds::Src& src ) ;
  virtual void levelEnd ( const Pds::Src& src ) ;

  // visit the data object
  virtual void dataObject ( const Pds::WaveformV1& data, const Pds::Src& src ) ;
  virtual void dataObject ( const Pds::Acqiris::ConfigV1& data, const Pds::Src& src ) ;

protected:

private:

  // Data members
  std::string m_fileName ;
  nexuspp::NxppFile* m_file ;
  std::set<std::string> m_existingGroups ;

  // Copy constructor and assignment are disabled by default
  O2ONexusWriter ( const O2ONexusWriter& ) ;
  O2ONexusWriter operator = ( const O2ONexusWriter& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2ONEXUSWRITER_H
