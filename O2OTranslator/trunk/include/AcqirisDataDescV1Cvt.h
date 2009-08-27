#ifndef O2OTRANSLATOR_ACQIRISDATADESCV1CVT_H
#define O2OTRANSLATOR_ACQIRISDATADESCV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisDataDescV1Cvt.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/DataTypeCvtI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/acqiris/ConfigV1.hh"
#include "pdsdata/acqiris/DataDescV1.hh"
#include "H5DataTypes/ObjectContainer.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Special converter class for Pds::Acqiris::DataDescV1
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

class AcqirisDataDescV1Cvt : public DataTypeCvtI {
public:

  // Default constructor
  AcqirisDataDescV1Cvt ( hdf5pp::Group group,
                         hsize_t chunk_size,
                         int deflate ) ;

  // Destructor
  virtual ~AcqirisDataDescV1Cvt () ;

  /// main method of this class
  virtual void convert ( const void* data,
                         const Pds::TypeId& typeId,
                         const Pds::DetInfo& detInfo,
                         const H5DataTypes::XtcClockTime& time ) ;

  void setGroup( hdf5pp::Group group ) ;

protected:

private:

  typedef H5DataTypes::ObjectContainer<uint64_t> TimestampCont ;
  typedef H5DataTypes::ObjectContainer<uint16_t> WaveformCont ;
  typedef H5DataTypes::ObjectContainer<H5DataTypes::XtcClockTime> XtcClockTimeCont ;

  // Data members
  hdf5pp::Group m_group ;
  hsize_t m_chunk_size ;
  int m_deflate ;
  const Pds::Acqiris::ConfigV1* m_config ;
  hdf5pp::Type m_tsType ;
  hdf5pp::Type m_wfType ;
  TimestampCont* m_timestampCont ;
  WaveformCont* m_waveformCont ;
  XtcClockTimeCont* m_timeCont ;

  // Copy constructor and assignment are disabled by default
  AcqirisDataDescV1Cvt ( const AcqirisDataDescV1Cvt& ) ;
  AcqirisDataDescV1Cvt& operator = ( const AcqirisDataDescV1Cvt& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_ACQIRISDATADESCV1CVT_H
