#ifndef O2OTRANSLATOR_BLDDATAPIMV1CVT_H
#define O2OTRANSLATOR_BLDDATAPIMV1CVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class BldDataPimV1Cvt.
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
#include "H5DataTypes/BldDataPimV1.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------
#include "O2OTranslator/CvtOptions.h"

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Special converter class for Pds::BldDataPimV1
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

class BldDataPimV1Cvt : public EvtDataTypeCvt<Pds::BldDataPimV1> {
public:

  typedef Pds::BldDataPimV1 XtcType;
  typedef H5DataTypes::BldDataPimV1 H5Type;

  // constructor takes a location where the data will be stored
  BldDataPimV1Cvt(const hdf5pp::Group& group, const std::string& typeGroupName,
      const Pds::Src& src, const CvtOptions& cvtOptions, int schemaVersion) ;

  // Destructor
  virtual ~BldDataPimV1Cvt () ;

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
  typedef H5DataTypes::ObjectContainer<const unsigned char> ImageCont ;

  // Data members
  hdf5pp::Type m_imgType ;
  boost::shared_ptr<DataCont> m_dataCont ;
  boost::shared_ptr<ImageCont> m_imageCont ;
  size_t n_miss;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_BLDDATAPIMV1CVT_H
