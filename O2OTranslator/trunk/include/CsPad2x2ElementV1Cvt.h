#ifndef O2OTRANSLATOR_CSPAD2X2ELEMENTV1CVT_H
#define O2OTRANSLATOR_CSPAD2X2ELEMENTV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CsPad2x2ElementV1Cvt.
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
#include "H5DataTypes/CsPad2x2ElementV1.h"
#include "O2OTranslator/CalibObjectStore.h"
#include "O2OTranslator/ConfigObjectStore.h"
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

/**
 *  Special converter class for Pds::CsPad::ElementV1 XTC class
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

class CsPad2x2ElementV1Cvt : public EvtDataTypeCvt<Pds::CsPad2x2::ElementV1> {
public:

  typedef Pds::CsPad2x2::ElementV1 XtcType ;

  // constructor
  CsPad2x2ElementV1Cvt ( const std::string& typeGroupName,
                      const CalibObjectStore& calibStore,
                      hsize_t chunk_size,
                      int deflate ) ;

  // Destructor
  virtual ~CsPad2x2ElementV1Cvt () ;

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

  typedef CvtDataContainer<CvtDataContFactoryTyped<H5DataTypes::CsPad2x2ElementV1> > ElementCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<int16_t> > PixelDataCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<float> > CommonModeDataCont ;
  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::XtcClockTime> > XtcClockTimeCont ;

  // Data members
  const CalibObjectStore& m_calibStore;
  hsize_t m_chunk_size ;
  int m_deflate ;
  ElementCont* m_elementCont ;
  PixelDataCont* m_pixelDataCont ;
  CommonModeDataCont* m_cmodeDataCont ;
  XtcClockTimeCont* m_timeCont ;

  // Copy constructor and assignment are disabled by default
  CsPad2x2ElementV1Cvt ( const CsPad2x2ElementV1Cvt& ) ;
  CsPad2x2ElementV1Cvt& operator = ( const CsPad2x2ElementV1Cvt& ) ;
};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CSPAD2X2ELEMENTV1CVT_H
