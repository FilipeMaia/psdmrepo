#ifndef O2OTRANSLATOR_OCEANOPTICSDATAV1CVT_H
#define O2OTRANSLATOR_OCEANOPTICSDATAV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OceanOpticsDataV1Cvt.
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
#include "H5DataTypes/OceanOpticsDataV1.h"
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
 *  @brief Special converter class for Pds::OceanOptics::DataV1 XTC class
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class OceanOpticsDataV1Cvt : public EvtDataTypeCvt<Pds::OceanOptics::DataV1> {
public:

  typedef Pds::OceanOptics::DataV1 XtcType ;
  typedef H5DataTypes::OceanOpticsDataV1 H5Type ;

  // Default constructor
  OceanOpticsDataV1Cvt(const hdf5pp::Group& group,
                       const std::string& typeGroupName,
                       const Pds::Src& src,
                       const ConfigObjectStore& configStore,
                       const CvtOptions& cvtOptions,
                       int schemaVersion);

  // Destructor
  virtual ~OceanOpticsDataV1Cvt () ;

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

  typedef H5DataTypes::ObjectContainer<H5Type> ObjectCont ;
  typedef H5DataTypes::ObjectContainer<uint16_t> DataCont ;
  typedef H5DataTypes::ObjectContainer<float> CorrectedDataCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  boost::shared_ptr<ObjectCont> m_objCont ;
  boost::shared_ptr<DataCont> m_dataCont ;
  boost::shared_ptr<CorrectedDataCont> m_corrDataCont ;
  size_t n_miss;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_OCEANOPTICSDATAV1CVT_H
