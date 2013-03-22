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
#include "O2OTranslator/CvtOptions.h"

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
  typedef H5DataTypes::PnCCDFrameV1 H5Type ;

  // constructor
  PnCCDFrameV1Cvt ( const hdf5pp::Group& group,
                    const std::string& typeGroupName,
                    const Pds::Src& src,
                    const ConfigObjectStore& configStore,
                    const CvtOptions& cvtOptions,
                    int schemaVersion ) ;

  // Destructor
  virtual ~PnCCDFrameV1Cvt () ;

protected:

  /// method called to create all necessary data containers
  virtual void makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src);

  // typed conversion method
  virtual void fillContainers(hdf5pp::Group group,
                              const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src);

  // fill containers for missing data
  virtual void fillMissing(hdf5pp::Group group,
                           const Pds::TypeId& typeId,
                           const O2OXtcSrc& src);

private:

  typedef H5DataTypes::ObjectContainer<H5Type> FrameCont ;
  typedef H5DataTypes::ObjectContainer<uint16_t> FrameDataCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  boost::shared_ptr<FrameCont> m_frameCont ;
  boost::shared_ptr<FrameDataCont> m_frameDataCont ;
  size_t n_miss;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_PNCCDFRAMEV1CVT_H
