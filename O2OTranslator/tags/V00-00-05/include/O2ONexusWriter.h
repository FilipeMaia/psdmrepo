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
#include <map>

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

class O2OFileNameFactory ;

class O2ONexusWriter : public O2OXtcScannerI {
public:

  enum State { Undefined, Mapped, Configured, Running } ;

  // Default constructor
  O2ONexusWriter ( const O2OFileNameFactory& nameFactory ) ;

  // Destructor
  virtual ~O2ONexusWriter () ;

  // signal start/end of the event (datagram)
  virtual void eventStart ( const Pds::Sequence& seq ) ;
  virtual void eventEnd ( const Pds::Sequence& seq ) ;

  // signal start/end of the level
  virtual void levelStart ( const Pds::Src& src ) ;
  virtual void levelEnd ( const Pds::Src& src ) ;

  // visit the data object
  virtual void dataObject ( const Pds::Acqiris::ConfigV1& data, const Pds::Src& src ) ;
  virtual void dataObject ( const Pds::Acqiris::DataDescV1& data, const Pds::Src& src ) ;
  virtual void dataObject ( const Pds::Camera::FrameFexConfigV1& data, const Pds::Src& src ) ;
  virtual void dataObject ( const Pds::Camera::FrameV1& data, const Pds::Src& src ) ;
  virtual void dataObject ( const Pds::Camera::TwoDGaussianV1& data, const Pds::Src& src ) ;
  virtual void dataObject ( const Pds::EvrData::ConfigV1& data, const Pds::Src& src ) ;
  virtual void dataObject ( const Pds::Opal1k::ConfigV1& data, const Pds::Src& src ) ;

protected:

private:

  struct CmpDetInfo {
    bool operator()( const Pds::DetInfo& lhs, const Pds::DetInfo& rhs ) const ;
  };

  typedef std::map<Pds::DetInfo,Pds::Acqiris::ConfigV1, CmpDetInfo> AcqConfigMap ;

  // Data members
  const O2OFileNameFactory& m_nameFactory ;
  nexuspp::NxppFile* m_file ;
  State m_state ;
  AcqConfigMap m_acqConfigMap ;

  // Copy constructor and assignment are disabled by default
  O2ONexusWriter ( const O2ONexusWriter& ) ;
  O2ONexusWriter operator = ( const O2ONexusWriter& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_O2ONEXUSWRITER_H
