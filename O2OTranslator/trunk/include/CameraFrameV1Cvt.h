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
#include "O2OTranslator/DataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "H5DataTypes/CameraFrameV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "H5DataTypes/ObjectContainer.h"

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

class CameraFrameV1Cvt : public DataTypeCvt<Pds::Camera::FrameV1> {
public:

  // constructor takes a location where the data will be stored
  CameraFrameV1Cvt ( hdf5pp::Group group,
                     hsize_t chunk_size,
                     int deflate ) ;

  // Destructor
  virtual ~CameraFrameV1Cvt () ;

  // typed conversion method
  virtual void typedConvert ( const Pds::Camera::FrameV1& data,
                              const H5DataTypes::XtcClockTime& time ) ;

protected:

private:

  typedef H5DataTypes::ObjectContainer<H5DataTypes::XtcClockTime> XtcClockTimeCont ;
  typedef H5DataTypes::ObjectContainer<H5DataTypes::CameraFrameV1> DataCont ;
  typedef H5DataTypes::ObjectContainer<const unsigned char> ImageCont ;

  // Data members
  hdf5pp::Group m_group ;
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
