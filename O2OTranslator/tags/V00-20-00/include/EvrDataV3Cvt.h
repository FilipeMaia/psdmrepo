#ifndef O2OTRANSLATOR_EVRDATAV3CVT_H
#define O2OTRANSLATOR_EVRDATAV3CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvrDataV3Cvt.
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
#include "H5DataTypes/EvrDataV3.h"
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
 *  Special converter class for Pds::EvrData::DataV3 XTC class
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */
class EvrDataV3Cvt : public EvtDataTypeCvt<Pds::EvrData::DataV3> {
public:

  typedef Pds::EvrData::DataV3 XtcType ;

  // constructor
  EvrDataV3Cvt ( const std::string& typeGroupName,
                 const ConfigObjectStore& configStore,
                 hsize_t chunk_size,
                 int deflate ) ;

  // Destructor
  virtual ~EvrDataV3Cvt () ;

protected:

  // typed conversion method
  virtual void typedConvertSubgroup ( hdf5pp::Group group,
                                      const XtcType& data,
                                      size_t size,
                                      const Pds::TypeId& typeId,
                                      const O2OXtcSrc& src,
                                      const H5DataTypes::XtcClockTimeStamp& time ) ;

  /// method called when the driver closes a group in the file
  virtual void closeSubgroup( hdf5pp::Group group ) ;

private:

  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::EvrDataV3> > EvrDataCont ;
  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::XtcClockTimeStamp> > XtcClockTimeCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  hsize_t m_chunk_size ;
  int m_deflate ;
  EvrDataCont* m_evrDataCont ;
  XtcClockTimeCont* m_timeCont ;

};


} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EVRDATAV3CVT_H
