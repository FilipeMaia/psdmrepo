#ifndef O2OTRANSLATOR_EVTDATATYPECVT_H
#define O2OTRANSLATOR_EVTDATATYPECVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class EvtDataTypeCvt.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <map>
#include <stack>
#include <boost/lexical_cast.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/DataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "H5DataTypes/ObjectContainer.h"
#include "H5DataTypes/XtcDamage.h"
#include "H5DataTypes/XtcClockTimeStamp.h"
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/CvtOptions.h"
#include "O2OTranslator/O2OXtcSrc.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace O2OTranslator {

/**
 *  Data converter class for event-type data objects
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

template <typename XtcType>
class EvtDataTypeCvt : public DataTypeCvt<XtcType> {
public:

  /**
   *  Constructor for converter
   *
   *  @param[in] group          HDF5 group inside which we create our stuff
   *  @param[in] typeGroupName  Name of the group for this type, arbitrary string usually
   *                            derived from type, should be unique.
   *  @param[in] src            Data source
   *  @param[in] cvtOptions     Object holding conversion options
   */
  EvtDataTypeCvt(hdf5pp::Group group, const std::string& typeGroupName, const Pds::Src& src, const CvtOptions& cvtOptions)
    : DataTypeCvt<XtcType>()
    , m_typeGroupName(typeGroupName)
    , m_cvtOptions(cvtOptions)
    , m_group()
    , m_timeCont(0)
    , m_damageCont(0)
    , m_maskCont(0)
  {
    // check if the group already there
    const std::string& srcName = boost::lexical_cast<std::string>(src);
    if ( group.hasChild(typeGroupName) ) {
      hdf5pp::Group typeGroup = group.openGroup(typeGroupName);
      if (typeGroup.hasChild(srcName)) {
        m_group = typeGroup.openGroup(srcName);
        MsgLog("ConfigDataTypeCvt", trace, "group " << typeGroupName << '/' << srcName << " already exists") ;
        return;
      }
    }

    // create new group
    m_group = group.createGroup(typeGroupName + "/" + srcName);
  }

  // Destructor
  virtual ~EvtDataTypeCvt ()
  {
    delete m_timeCont;
    delete m_damageCont;
    delete m_maskCont;
  }

  // typed conversion method
  virtual void typedConvert ( const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src,
                              const H5DataTypes::XtcClockTimeStamp& time,
                              Pds::Damage damage )
  {
    // initialize all containers
    if (not m_timeCont) {

      // make container for time
      m_timeCont = makeCont<XtcClockTimeCont>( "time", m_group, true);

      // make container for damage array if requested
      if (m_cvtOptions.storeDamage()) {
        m_damageCont = makeCont<XtcDamageCont>("_damage", m_group, false);
      }

      // make container for mask array if needed
      if (m_cvtOptions.fillMissing()) {
        m_maskCont = makeCont<MaskCont>("_mask",  m_group, false);
      }

      // call subclass method to make container for data objects
      makeContainers(m_group, typeId, src);

    }

    // fill time container with data
    m_timeCont->append(time);
    if (m_damageCont) m_damageCont->append(damage);
    if (m_maskCont) m_maskCont->append(uint8_t(damage.bits() == 0));

    // call subclass method to fill its containers with data
    this->fillContainers(m_group, data, size, typeId, src);
  }

  // method called to fill void spaces for missing data
  virtual void fillMissing(const Pds::TypeId& typeId,
                           const O2OXtcSrc& src,
                           const H5DataTypes::XtcClockTimeStamp& time,
                           Pds::Damage damage)
  {
    // TODO: add implementation here
  }

  const std::string& typeGroupName() const { return m_typeGroupName ; }
  
protected:

  // create container of given type
  template <typename ContType>
  ContType* makeCont(const std::string& name, hdf5pp::Group& location, bool shuffle,
      hdf5pp::Type type = hdf5pp::TypeTraits<typename ContType::value_type>::stored_type())
  {
    return new ContType(name, location, type, m_cvtOptions.chunkSize(), m_cvtOptions.compLevel(), shuffle);
  }

  /// method called to create all necessary data containers
  virtual void makeContainers(hdf5pp::Group group, const Pds::TypeId& typeId, const O2OXtcSrc& src) = 0;

  // typed conversion method
  virtual void fillContainers(hdf5pp::Group group,
                              const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src) = 0 ;

private:

  typedef H5DataTypes::ObjectContainer<H5DataTypes::XtcClockTimeStamp> XtcClockTimeCont ;
  typedef H5DataTypes::ObjectContainer<H5DataTypes::XtcDamage> XtcDamageCont ;
  typedef H5DataTypes::ObjectContainer<uint8_t> MaskCont ;

  // Data members
  const std::string m_typeGroupName ;  ///< Group name for this type
  const CvtOptions m_cvtOptions;
  hdf5pp::Group m_group;
  XtcClockTimeCont* m_timeCont ;
  XtcDamageCont* m_damageCont;
  MaskCont* m_maskCont;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EVTDATATYPECVT_H
