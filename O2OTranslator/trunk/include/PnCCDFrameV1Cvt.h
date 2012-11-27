#ifndef O2OTRANSLATOR_PNCCDFRAMEV1CVT_H
#define O2OTRANSLATOR_PNCCDFRAMEV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class PnCCDFrameV1Cvt.
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
#include "H5DataTypes/PnCCDFrameV1.h"
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
 *  Special converter class for Pds::PNCCD::FrameV1 XTC class
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

class PnCCDFrameV1Cvt : public EvtDataTypeCvt<Pds::PNCCD::FrameV1> {
public:

  typedef Pds::PNCCD::FrameV1 XtcType ;

  // constructor
  PnCCDFrameV1Cvt ( const std::string& typeGroupName,
                    const ConfigObjectStore& configStore,
                    hsize_t chunk_size,
                    int deflate ) ;

  // Destructor
  virtual ~PnCCDFrameV1Cvt () ;

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

  typedef CvtDataContainer<CvtDataContFactoryTyped<H5DataTypes::PnCCDFrameV1> > FrameCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<uint16_t> > FrameDataCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  FrameCont* m_frameCont ;
  FrameDataCont* m_frameDataCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_PNCCDFRAMEV1CVT_H
