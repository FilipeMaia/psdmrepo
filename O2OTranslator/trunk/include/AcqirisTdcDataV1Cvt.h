#ifndef O2OTRANSLATOR_ACQIRISTDCDATAV1CVT_H
#define O2OTRANSLATOR_ACQIRISTDCDATAV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class AcqirisTdcDataV1Cvt.
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
#include "H5DataTypes/AcqirisTdcDataV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "O2OTranslator/CvtDataContainer.h"
#include "O2OTranslator/CvtDataContFactoryDef.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Special converter class for Pds::Acqiris::TdcDataV1
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

class AcqirisTdcDataV1Cvt : public EvtDataTypeCvt<Pds::Acqiris::TdcDataV1> {
public:

  typedef H5DataTypes::AcqirisTdcDataV1 H5Type ;
  typedef Pds::Acqiris::TdcDataV1 XtcType ;

  // constructor takes a location where the data will be stored
  AcqirisTdcDataV1Cvt (const std::string& typeGroupName,
                       hsize_t chunk_size,
                       int deflate) ;

  // Destructor
  virtual ~AcqirisTdcDataV1Cvt () ;

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

  // Data members
  DataCont* m_dataCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_ACQIRISTDCDATAV1CVT_H
