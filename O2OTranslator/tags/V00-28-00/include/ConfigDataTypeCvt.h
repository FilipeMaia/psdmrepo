#ifndef O2OTRANSLATOR_CONFIGDATATYPECVT_H
#define O2OTRANSLATOR_CONFIGDATATYPECVT_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class ConfigDataTypeCvt.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <cassert>
#include <boost/lexical_cast.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/DataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"
#include "O2OTranslator/O2OXtcSrc.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
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

namespace O2OTranslator {

template <typename H5Type>
class ConfigDataTypeCvt : public DataTypeCvt<typename H5Type::XtcType> {
public:

  typedef typename H5Type::XtcType XtcType ;

  /**
   *  Constructor for converter.
   *
   *  @param[in] group          HDF5 group inside which we create our stuff
   *  @param[in] typeGroupName  Name of the group for this type, arbitrary string usually
   *                            derived from type, should be unique.
   *  @param[in] src            Data source
   */
  ConfigDataTypeCvt(hdf5pp::Group group, const std::string& typeGroupName,
      const Pds::Src& src, int schemaVersion)
    : DataTypeCvt<typename H5Type::XtcType>()
    , m_group(group)
    , m_typeGroupName(typeGroupName)
    , m_src(src)
    , m_schemaVersion(schemaVersion)
  {
  }

  // Destructor
  virtual ~ConfigDataTypeCvt () {}

  // typed conversion method
  virtual void typedConvert ( const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src,
                              const H5DataTypes::XtcClockTimeStamp& time,
                              Pds::Damage damage )
  {
    // check data size
    if (H5Type::xtcSize(data) != size) {
      if ( size == 0 ) {
        MsgLog("ConfigDataTypeCvt", warning, "Zero XTC payload in " << m_typeGroupName << ", expected size " <<H5Type::xtcSize(data)) ;
        return;
      }
      throw O2OXTCSizeException(ERR_LOC, m_typeGroupName, H5Type::xtcSize(data), size);
    }
    
    // check if the group already there
    const std::string& srcName = boost::lexical_cast<std::string>(m_src);
    if (m_group.hasChild(m_typeGroupName)) {
      hdf5pp::Group typeGroup = m_group.openGroup(m_typeGroupName);
      if (typeGroup.hasChild(srcName)) {
        MsgLog("ConfigDataTypeCvt", trace, "group " << m_typeGroupName << '/' << srcName << " already exists") ;
        return;
      }
    }

    // create separate group
    hdf5pp::Group group = m_group.createGroup(m_typeGroupName + "/" + srcName);

    // store some group attributes
    uint64_t srcVal = (uint64_t(m_src.phy()) << 32) + m_src.log();
    group.createAttr<uint64_t>("_xtcSrc").store(srcVal);
    group.createAttr<uint32_t>("_schemaVersion").store(m_schemaVersion);

    // store the data
    H5Type::store(data, group);
  }

  // method called to fill void spaces for missing data
  virtual void missingConvert(const Pds::TypeId& typeId,
                              const O2OXtcSrc& src,
                              const H5DataTypes::XtcClockTimeStamp& time,
                              Pds::Damage damage)
  {
    // For configuration objects we do not do anything if data is missing/damaged
  }

protected:

private:

  // Data members
  hdf5pp::Group m_group;
  std::string m_typeGroupName;
  Pds::Src m_src;
  int m_schemaVersion;
};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CONFIGDATATYPECVT_H
