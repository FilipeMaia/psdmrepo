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
#include "O2OTranslator/CvtOptions.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/// @addtogroup O2OTranslator

/**
 *  @ingroup O2OTranslator
 *
 *  Special converter class for Pds::Timepix::DataV2 XTC class
 *
 *  This software was developed for the LCLS project.  If you use all or
 *  part of it, please give an appropriate acknowledgment.
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
  TimepixDataV2Cvt ( const hdf5pp::Group& group,
      const std::string& typeGroupName,
      const Pds::Src& src,
      const CvtOptions& cvtOptions,
      int schemaVersion ) ;

  // Destructor
  virtual ~TimepixDataV2Cvt () ;

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
  typedef H5DataTypes::ObjectContainer<uint16_t> ImageCont ;

  // Data members
  boost::shared_ptr<DataCont> m_dataCont ;
  boost::shared_ptr<ImageCont> m_imageCont ;
  size_t n_miss;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_TIMEPIXDATAV2CVT_H
