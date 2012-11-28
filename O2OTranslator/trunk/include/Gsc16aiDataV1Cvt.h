#ifndef O2OTRANSLATOR_GSC16AIDATAV1CVT_H
#define O2OTRANSLATOR_GSC16AIDATAV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Gsc16aiDataV1Cvt.
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
#include "H5DataTypes/Gsc16aiDataV1.h"
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
 *  Special converter class for Pds::Gsc16ai::DataV1 XTC class
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class Gsc16aiDataV1Cvt : public EvtDataTypeCvt<Pds::Gsc16ai::DataV1> {
public:

  typedef Pds::Gsc16ai::DataV1 XtcType;
  typedef H5DataTypes::Gsc16aiDataV1 H5Type;
  
  // constructor
  Gsc16aiDataV1Cvt(const std::string& typeGroupName,
      const ConfigObjectStore& configStore,
      hsize_t chunk_size,
      int deflate);

  // Destructor
  virtual ~Gsc16aiDataV1Cvt () ;

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
  typedef CvtDataContainer<CvtDataContFactoryTyped<uint16_t> > ValueCont ;

  // Data members
  const ConfigObjectStore& m_configStore;
  DataCont* m_dataCont ;
  ValueCont* m_valueCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_GSC16AIDATAV1CVT_H
