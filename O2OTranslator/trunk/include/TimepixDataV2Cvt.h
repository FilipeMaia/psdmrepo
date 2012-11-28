#ifndef O2OTRANSLATOR_TIMEPIXDATAV2CVT_H
#define O2OTRANSLATOR_TIMEPIXDATAV2CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class TimepixDataV2Cvt.
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
#include "H5DataTypes/TimepixDataV2.h"
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
 *  Special converter class for Pds::Timepix::DataV2 XTC class
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

class TimepixDataV2Cvt : public EvtDataTypeCvt<Pds::Timepix::DataV2> {
public:

  typedef Pds::Timepix::DataV2 XtcType ;
  typedef H5DataTypes::TimepixDataV2 H5Type ;

  // constructor
  TimepixDataV2Cvt ( const std::string& typeGroupName,
                     hsize_t chunk_size,
                     int deflate ) ;

  // Destructor
  virtual ~TimepixDataV2Cvt () ;

protected:

  /// method called to create all necessary data containers
  virtual void makeContainers(hsize_t chunk_size, int deflate,
      const Pds::TypeId& typeId, const O2OXtcSrc& src);

  // typed conversion method
  virtual void fillContainers(hdf5pp::Group group,
                              const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src);

  /// method called when the driver closes a group in the file
  virtual void closeContainers(hdf5pp::Group group);

private:

  typedef CvtDataContainer<CvtDataContFactoryDef<H5Type> > DataCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<uint16_t> > ImageCont ;

  // Data members
  DataCont* m_dataCont ;
  ImageCont* m_imageCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_TIMEPIXDATAV2CVT_H
