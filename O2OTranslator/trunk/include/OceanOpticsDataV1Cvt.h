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
  OceanOpticsDataV1Cvt(const std::string& typeGroupName,
                       const ConfigObjectStore& configStore,
                       hsize_t chunk_size,
                       int deflate);

  // Destructor
  virtual ~OceanOpticsDataV1Cvt () ;

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

  typedef CvtDataContainer<CvtDataContFactoryDef<H5Type> > ObjectCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<uint16_t> > DataCont ;
  typedef CvtDataContainer<CvtDataContFactoryTyped<float> > CorrectedDataCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  ObjectCont* m_objCont ;
  DataCont* m_dataCont ;
  CorrectedDataCont* m_corrDataCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_OCEANOPTICSDATAV1CVT_H
