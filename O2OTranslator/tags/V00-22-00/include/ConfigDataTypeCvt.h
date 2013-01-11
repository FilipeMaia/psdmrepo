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
#include <stack>
#include <cassert>

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/DataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/O2OExceptions.h"
#include "O2OTranslator/SrcFilter.h"

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
   *  @param[in] typeGroupName  Name of the group for this type, arbitrary string usually
   *                            derived from type, should be unique.
   *  @param[in] srcFilter      Source filter object, default is to allow all sources
   */
  ConfigDataTypeCvt ( const std::string& typeGroupName, SrcFilter srcFilter = SrcFilter() )
    : DataTypeCvt<typename H5Type::XtcType>()
    , m_typeGroupName(typeGroupName)
    , m_srcFilter(srcFilter)
    , m_groups()
  {}

  // Destructor
  virtual ~ConfigDataTypeCvt () {}

  // typed conversion method
  virtual void typedConvert ( const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src,
                              const H5DataTypes::XtcClockTimeStamp& time )
  {
    // filter source
    if (not m_srcFilter(src.top())) return;

    // this should not happen
    assert ( not m_groups.empty() ) ;

    // check data size
    if ( H5Type::xtcSize(data) != size ) {
      if ( size == 0 ) {
        MsgLog("ConfigDataTypeCvt", warning, "Zero XTC payload in " << m_typeGroupName << ", expected size " <<H5Type::xtcSize(data)) ;
        return;
      }
      throw O2OXTCSizeException ( ERR_LOC, m_typeGroupName, H5Type::xtcSize(data), size ) ;
    }
    
    // get the name of the group for this object
    const std::string& grpName = m_typeGroupName + "/" + src.name() ;

    if ( m_groups.top().hasChild(m_typeGroupName) ) {
      hdf5pp::Group typeGroup = m_groups.top().openGroup(m_typeGroupName);
      if ( typeGroup.hasChild(src.name()) ) {
        MsgLog("ConfigDataTypeCvt", trace, "group " << grpName << " already exists") ;
        return;
      }
    }

    // create separate group
    hdf5pp::Group grp = m_groups.top().createGroup( grpName );

    // store the data
    H5Type::store ( data, grp ) ;
  }

  /// method called when the driver makes a new group in the file
  virtual void openGroup( hdf5pp::Group group ) {
    m_groups.push ( group ) ;
  }

  /// method called when the driver closes a group in the file
  virtual void closeGroup( hdf5pp::Group group ) {
    if ( m_groups.empty() ) return ;
    while ( m_groups.top() != group ) m_groups.pop() ;
    if ( m_groups.empty() ) return ;
    m_groups.pop() ;
  }

protected:

private:

  // Data members
  std::string m_typeGroupName ;
  const SrcFilter m_srcFilter;
  std::stack<hdf5pp::Group> m_groups ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CONFIGDATATYPECVT_H
