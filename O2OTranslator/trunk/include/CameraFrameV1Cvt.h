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
#include "O2OTranslator/CvtOptions.h"

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
  typedef H5DataTypes::CameraFrameV1 H5Type ;

  // constructor takes a location where the data will be stored
  CameraFrameV1Cvt ( const hdf5pp::Group& group, const std::string& typeGroupName,
      const Pds::Src& src, const CvtOptions& cvtOptions ) ;

  // Destructor
  virtual ~CameraFrameV1Cvt () ;

protected:

  /// method called to create all necessary data containers
  virtual void makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src);

  // typed conversion method
  virtual void fillContainers(hdf5pp::Group group,
                              const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src);

private:

  typedef H5DataTypes::ObjectContainer<H5Type> DataCont ;
  typedef H5DataTypes::ObjectContainer<const unsigned char> ImageCont ;
  typedef H5DataTypes::ObjectContainer<uint16_t> DimFixFlagCont ;

  // Data members
  hdf5pp::Type m_imgType ;
  DataCont* m_dataCont ;
  ImageCont* m_imageCont ;
  DimFixFlagCont* m_dimFixFlagCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CAMERAFRAMEV1CVT_H
