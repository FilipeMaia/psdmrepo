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
#include "O2OTranslator/EvtDataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/acqiris/DataDescV1.hh"
#include "O2OTranslator/CvtDataContainer.h"
#include "O2OTranslator/CvtDataContFactoryDef.h"
#include "O2OTranslator/CvtDataContFactoryTyped.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

class ConfigObjectStore;

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

class AcqirisDataDescV1Cvt : public EvtDataTypeCvt<Pds::Acqiris::DataDescV1> {
public:

  typedef Pds::Acqiris::DataDescV1 XtcType ;

  // Default constructor
  AcqirisDataDescV1Cvt ( const std::string& typeGroupName,
                         const ConfigObjectStore& configStore,
                         hsize_t chunk_size,
                         int deflate ) ;

  // Destructor
  virtual ~AcqirisDataDescV1Cvt () ;

protected:

  // typed conversion method
  virtual void typedConvertSubgroup ( hdf5pp::Group group,
                                      const XtcType& data,
                                      size_t size,
                                      const Pds::TypeId& typeId,
                                      const XtcInput::XtcSrcStack& src,
                                      const H5DataTypes::XtcClockTime& time ) ;

  /// method called when the driver closes a group in the file
  virtual void closeSubgroup( hdf5pp::Group group ) ;


private:

  typedef CvtDataContainer<CvtDataContFactoryTyped<uint64_t> > TimestampCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<int16_t> > WaveformCont ;
  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::XtcClockTime> > XtcClockTimeCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  hsize_t m_chunk_size ;
  int m_deflate ;
  TimestampCont* m_timestampCont ;
  WaveformCont* m_waveformCont ;
  XtcClockTimeCont* m_timeCont ;

  // Copy constructor and assignment are disabled by default
  AcqirisDataDescV1Cvt ( const AcqirisDataDescV1Cvt& ) ;
  AcqirisDataDescV1Cvt& operator = ( const AcqirisDataDescV1Cvt& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_ACQIRISDATADESCV1CVT_H
