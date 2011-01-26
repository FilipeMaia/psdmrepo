#ifndef O2OTRANSLATOR_ACQIRISTDCDATAV1CVT_H
#define O2OTRANSLATOR_ACQIRISTDCDATAV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisTdcDataV1Cvt.
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
#include "H5DataTypes/AcqirisTdcDataV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "O2OTranslator/CvtDataContainer.h"
#include "O2OTranslator/CvtDataContFactoryDef.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Special converter class for Pds::Acqiris::TdcDataV1
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

class AcqirisTdcDataV1Cvt : public EvtDataTypeCvt<Pds::Acqiris::TdcDataV1> {
public:

  typedef H5DataTypes::AcqirisTdcDataV1 H5Type ;
  typedef Pds::Acqiris::TdcDataV1 XtcType ;

  // constructor takes a location where the data will be stored
  AcqirisTdcDataV1Cvt (const std::string& typeGroupName,
                       hsize_t chunk_size,
                       int deflate) ;

  // Destructor
  virtual ~AcqirisTdcDataV1Cvt () ;

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

  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::XtcClockTime> > XtcClockTimeCont ;
  typedef CvtDataContainer<CvtDataContFactoryDef<H5Type> > DataCont ;

  // Data members
  hsize_t m_chunk_size ;
  int m_deflate ;
  DataCont* m_dataCont ;
  XtcClockTimeCont* m_timeCont ;

  // Copy constructor and assignment are disabled by default
  AcqirisTdcDataV1Cvt ( const AcqirisTdcDataV1Cvt& ) ;
  AcqirisTdcDataV1Cvt& operator = ( const AcqirisTdcDataV1Cvt& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_ACQIRISTDCDATAV1CVT_H
