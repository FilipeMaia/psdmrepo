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
#include "O2OTranslator/CvtOptions.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

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
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

class AcqirisTdcDataV1Cvt : public EvtDataTypeCvt<Pds::Acqiris::TdcDataV1> {
public:

  typedef Pds::Acqiris::TdcDataV1 XtcType ;
  typedef H5DataTypes::AcqirisTdcDataV1 H5Type ;

  // constructor takes a location where the data will be stored
  AcqirisTdcDataV1Cvt (const hdf5pp::Group& group, const std::string& typeGroupName,
      const Pds::Src& src, const CvtOptions& cvtOptions) ;

  // Destructor
  virtual ~AcqirisTdcDataV1Cvt () ;

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

  typedef H5DataTypes::ObjectContainer<H5Type> DataCont ;

  // Data members
  boost::shared_ptr<DataCont> m_dataCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_ACQIRISTDCDATAV1CVT_H
