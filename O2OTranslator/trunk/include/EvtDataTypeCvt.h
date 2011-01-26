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

//----------------------
// Base Class Headers --
//----------------------
#include "O2OTranslator/DataTypeCvt.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Group.h"
#include "O2OTranslator/CvtGroupMap.h"

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

  // constructor takes a location where the data will be stored
  EvtDataTypeCvt ( const std::string& typeGroupName )
    : DataTypeCvt<XtcType>()
    , m_typeGroupName(typeGroupName)
  {
  }

  // Destructor
  virtual ~EvtDataTypeCvt ()
  {
  }

  // typed conversion method
  virtual void typedConvert ( const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const XtcInput::XtcSrcStack& src,
                              const H5DataTypes::XtcClockTime& time )
  {
    hdf5pp::Group group = m_groups.top() ;
    hdf5pp::Group subgroup = m_group2group.find ( group, src.top() ) ;
    if ( not subgroup.valid() ) {

      // get the name of the group for this object
      const std::string& grpName = m_typeGroupName + "/" + src.name() ;

      // create separate group
      subgroup = group.createGroup( grpName );

      m_group2group.insert ( group, src.top(), subgroup ) ;
    }

    // call overloaded method and pass all data
    this->typedConvertSubgroup ( subgroup, data, size, typeId, src, time ) ;
  }

  /// method called when the driver makes a new group in the file
  virtual void openGroup( hdf5pp::Group group ) {
    m_groups.push ( group ) ;
  }

  /// method called when the driver closes a group in the file
  virtual void closeGroup( hdf5pp::Group group )
  {
    // tell my subobjects that we are closing all subgroups
    const CvtGroupMap::GroupList& subgroups = m_group2group.groups( group ) ;
    for ( CvtGroupMap::GroupList::const_iterator it = subgroups.begin() ; it != subgroups.end() ; ++ it ) {
      this->closeSubgroup( *it );
    }

    // remove it from the map
    m_group2group.erase( group ) ;

    // remove it from the stack
    if ( m_groups.empty() ) return ;
    while ( m_groups.top() != group ) m_groups.pop() ;
    if ( m_groups.empty() ) return ;
    m_groups.pop() ;
  }

  const std::string& typeGroupName() const { return m_typeGroupName ; }
  
protected:

  // typed conversion method
  virtual void typedConvertSubgroup ( hdf5pp::Group group,
                                      const XtcType& data,
                                      size_t size,
                                      const Pds::TypeId& typeId,
                                      const XtcInput::XtcSrcStack& src,
                                      const H5DataTypes::XtcClockTime& time ) = 0 ;

  /// method called when the driver closes a group in the file
  virtual void closeSubgroup( hdf5pp::Group group ) = 0 ;

private:

  typedef std::map<hdf5pp::Group,hdf5pp::Group> Group2Group ;

  // Data members
  std::string m_typeGroupName ;
  std::stack<hdf5pp::Group> m_groups ;
  CvtGroupMap m_group2group ;

  // Copy constructor and assignment are disabled by default
  EvtDataTypeCvt ( const EvtDataTypeCvt& ) ;
  EvtDataTypeCvt operator = ( const EvtDataTypeCvt& ) ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EVTDATATYPECVT_H
