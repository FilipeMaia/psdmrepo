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
#include <map>

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/EvtDataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "pdsdata/acqiris/ConfigV1.hh"
#include "pdsdata/acqiris/DataDescV1.hh"
#include "O2OTranslator/CvtDataContainer.h"
#include "O2OTranslator/CvtDataContFactoryDef.h"
#include "O2OTranslator/CvtDataContFactoryAcqirisV1.h"

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

class AcqirisDataDescV1Cvt : public EvtDataTypeCvt<Pds::Acqiris::DataDescV1> {
public:

  typedef Pds::Acqiris::DataDescV1 XtcType ;
  typedef Pds::Acqiris::ConfigV1 ConfigXtcType ;

  // Default constructor
  AcqirisDataDescV1Cvt ( const std::string& typeGroupName,
                         hsize_t chunk_size,
                         int deflate ) ;

  // Destructor
  virtual ~AcqirisDataDescV1Cvt () ;

  /// override base class method because we expect multiple types
  virtual void convert ( const void* data,
                         size_t size,
                         const Pds::TypeId& typeId,
                         const O2OXtcSrc& src,
                         const H5DataTypes::XtcClockTime& time ) ;

protected:

  // typed conversion method
  virtual void typedConvertSubgroup ( hdf5pp::Group group,
                                      const XtcType& data,
                                      size_t size,
                                      const Pds::TypeId& typeId,
                                      const O2OXtcSrc& src,
                                      const H5DataTypes::XtcClockTime& time ) ;

  /// method called when the driver closes a group in the file
  virtual void closeSubgroup( hdf5pp::Group group ) ;


private:

  typedef CvtDataContainer<CvtDataContFactoryAcqirisV1<uint64_t> > TimestampCont ;
  typedef CvtDataContainer<CvtDataContFactoryAcqirisV1<uint16_t> > WaveformCont ;
  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::XtcClockTime> > XtcClockTimeCont ;

  // comparison operator for Src objects
  struct _SrcCmp {
    bool operator()( const Pds::Src& lhs, const Pds::Src& rhs ) const ;
  };

  typedef std::map<Pds::Src,Pds::Acqiris::ConfigV1,_SrcCmp> ConfigMap ;

  // Data members
  hsize_t m_chunk_size ;
  int m_deflate ;
  ConfigMap m_config ;
  TimestampCont* m_timestampCont ;
  WaveformCont* m_waveformCont ;
  XtcClockTimeCont* m_timeCont ;

  // Copy constructor and assignment are disabled by default
  AcqirisDataDescV1Cvt ( const AcqirisDataDescV1Cvt& ) ;
  AcqirisDataDescV1Cvt& operator = ( const AcqirisDataDescV1Cvt& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_ACQIRISDATADESCV1CVT_H
