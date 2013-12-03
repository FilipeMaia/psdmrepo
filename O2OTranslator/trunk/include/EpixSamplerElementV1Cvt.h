#ifndef O2OTRANSLATOR_EPIXSAMPLERELEMENTV1CVT_H
#define O2OTRANSLATOR_EPIXSAMPLERELEMENTV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EpixSamplerElementV1Cvt.
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
#include "H5DataTypes/EpixSamplerElementV1.h"
#include "O2OTranslator/CvtOptions.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

class ConfigObjectStore;

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  @brief Special converter class for Pds::EpixSampler::ElementV1 XTC class
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class EpixSamplerElementV1Cvt : public EvtDataTypeCvt<Pds::EpixSampler::ElementV1> {
public:

  typedef Pds::EpixSampler::ElementV1 XtcType;
  typedef H5DataTypes::EpixSamplerElementV1 H5Type;
  
  // constructor
  EpixSamplerElementV1Cvt(const hdf5pp::Group& group,
      const std::string& typeGroupName,
      const Pds::Src& src,
      const ConfigObjectStore& configStore,
      const CvtOptions& cvtOptions,
      int schemaVersion);

  // Destructor
  virtual ~EpixSamplerElementV1Cvt () ;

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

  // finds config object and gets number of channels and samples from it, returns empty vector if config object is not there
  std::vector<int> shape(const O2OXtcSrc& src);

  typedef H5DataTypes::ObjectContainer<H5Type> DataCont ;
  typedef H5DataTypes::ObjectContainer<uint16_t> FrameCont ;
  typedef H5DataTypes::ObjectContainer<uint16_t> TemperatureCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  boost::shared_ptr<DataCont> m_dataCont ;
  boost::shared_ptr<FrameCont> m_frameCont ;
  boost::shared_ptr<TemperatureCont> m_temperatureCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EPIXSAMPLERELEMENTV1CVT_H
