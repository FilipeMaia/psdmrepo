#ifndef O2OTRANSLATOR_PRINCETONFRAMEV1CVT_H
#define O2OTRANSLATOR_PRINCETONFRAMEV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PrincetonFrameV1Cvt.
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
#include "H5DataTypes/PrincetonFrameV1.h"
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
 *  Special converter class for Pds::Princeton::FrameV1 XTC class
 *
 *  This software was developed for the LCLS project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class PrincetonFrameV1Cvt : public EvtDataTypeCvt<Pds::Princeton::FrameV1> {
public:

  typedef Pds::Princeton::FrameV1 XtcType ;

  // constructor
  PrincetonFrameV1Cvt ( const std::string& typeGroupName,
                        const ConfigObjectStore& configStore,
                        hsize_t chunk_size,
                        int deflate ) ;

  // Destructor
  virtual ~PrincetonFrameV1Cvt () ;

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

  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::PrincetonFrameV1> > FrameCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<uint16_t> > FrameDataCont ;
  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::XtcClockTime> > XtcClockTimeCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  hsize_t m_chunk_size ;
  int m_deflate ;
  FrameCont* m_frameCont ;
  FrameDataCont* m_frameDataCont ;
  XtcClockTimeCont* m_timeCont ;

  // Copy constructor and assignment are disabled by default
  PrincetonFrameV1Cvt ( const PrincetonFrameV1Cvt& ) ;
  PrincetonFrameV1Cvt& operator = ( const PrincetonFrameV1Cvt& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_PRINCETONFRAMEV1CVT_H
