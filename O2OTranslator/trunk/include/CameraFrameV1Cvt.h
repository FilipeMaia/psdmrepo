#ifndef O2OTRANSLATOR_CAMERAFRAMEV1CVT_H
#define O2OTRANSLATOR_CAMERAFRAMEV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CameraFrameV1Cvt.
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
#include "H5DataTypes/CameraFrameV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "O2OTranslator/CvtDataContainer.h"
#include "O2OTranslator/CvtDataContFactoryDef.h"
#include "O2OTranslator/CvtDataContFactoryTyped.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Special converter class for Pds::Camera::FrameV1
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

class CameraFrameV1Cvt : public EvtDataTypeCvt<Pds::Camera::FrameV1> {
public:

  typedef Pds::Camera::FrameV1 XtcType ;

  // constructor takes a location where the data will be stored
  CameraFrameV1Cvt ( const std::string& typeGroupName,
                     hsize_t chunk_size,
                     int deflate ) ;

  // Destructor
  virtual ~CameraFrameV1Cvt () ;

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
  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::CameraFrameV1> > DataCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<const unsigned char> > ImageCont ;

  // Data members
  hsize_t m_chunk_size ;
  int m_deflate ;
  hdf5pp::Type m_imgType ;
  DataCont* m_dataCont ;
  ImageCont* m_imageCont ;
  XtcClockTimeCont* m_timeCont ;

  // Copy constructor and assignment are disabled by default
  CameraFrameV1Cvt ( const CameraFrameV1Cvt& ) ;
  CameraFrameV1Cvt& operator = ( const CameraFrameV1Cvt& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CAMERAFRAMEV1CVT_H
