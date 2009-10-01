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

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/DataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

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

  // constructor takes a location where the data will be stored
  ConfigDataTypeCvt ( const std::string& typeGroupName )
    : DataTypeCvt<typename H5Type::XtcType>()
    , m_typeGroupName(typeGroupName)
    , m_groups()
  {}

  // Destructor
  virtual ~ConfigDataTypeCvt () {}

  // typed conversion method
  virtual void typedConvert ( const XtcType& data,
                              const Pds::TypeId& typeId,
                              const Pds::DetInfo& detInfo,
                              const H5DataTypes::XtcClockTime& time )
  {
    // this should not happen
    if ( m_groups.empty() ) return ;

    // get the name of the group for this object
    const std::string& grpName = this->cvtGroupName( m_typeGroupName, detInfo ) ;

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
  std::stack<hdf5pp::Group> m_groups ;

  // Copy constructor and assignment are disabled by default
  ConfigDataTypeCvt ( const ConfigDataTypeCvt& ) ;
  ConfigDataTypeCvt operator = ( const ConfigDataTypeCvt& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_CONFIGDATATYPECVT_H
