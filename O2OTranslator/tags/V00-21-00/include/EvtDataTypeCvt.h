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
#include "MsgLogger/MsgLogger.h"
#include "O2OTranslator/CvtGroupMap.h"
#include "O2OTranslator/CvtDataContainer.h"
#include "O2OTranslator/CvtDataContFactoryDef.h"

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
  EvtDataTypeCvt(const std::string& typeGroupName, hsize_t chunk_size, int deflate)
    : DataTypeCvt<XtcType>()
    , m_typeGroupName(typeGroupName)
    , m_chunk_size(chunk_size)
    , m_deflate(deflate)
    , m_groups()
    , m_group2group()
    , m_timeCont(0)

  {
  }

  // Destructor
  virtual ~EvtDataTypeCvt ()
  {
    delete m_timeCont ;
  }

  // typed conversion method
  virtual void typedConvert ( const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src,
                              const H5DataTypes::XtcClockTimeStamp& time )
  {
    hdf5pp::Group group = m_groups.top() ;
    hdf5pp::Group subgroup = m_group2group.find ( group, src.top() ) ;
    if ( not subgroup.valid() ) {

      // get the name of the group for this object
      const std::string& grpName = m_typeGroupName + "/" + src.name() ;

      // create separate group
      if (group.hasChild(grpName)) {
        MsgLog("EvtDataTypeCvt", trace, "EvtDataTypeCvt -- existing group " << grpName ) ;
        subgroup = group.openGroup( grpName );
      } else {
        MsgLog("EvtDataTypeCvt", trace, "EvtDataTypeCvt -- creating group " << grpName ) ;
        subgroup = group.createGroup( grpName );
      }

      m_group2group.insert ( group, src.top(), subgroup ) ;
    }

    // initialize all containers
    if (not m_timeCont) {

      // call subclass method to make container for data objects
      makeContainers(m_chunk_size, m_deflate, typeId, src);

      // make container for time
      CvtDataContFactoryDef<H5DataTypes::XtcClockTimeStamp> timeContFactory ( "time", m_chunk_size, m_deflate, true ) ;
      m_timeCont = new XtcClockTimeCont ( timeContFactory ) ;
    }

    // fill time container with data
    m_timeCont->container(subgroup)->append(time);

    // call subclass method to fill its containers with data
    this->fillContainers(subgroup, data, size, typeId, src);
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
      if (m_timeCont) m_timeCont->closeGroup(*it);
      this->closeContainers(*it);
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

  /// method called to create all necessary data containers
  virtual void makeContainers(hsize_t chunk_size, int deflate,
      const Pds::TypeId& typeId, const O2OXtcSrc& src) = 0;

  // typed conversion method
  virtual void fillContainers(hdf5pp::Group group,
                              const XtcType& data,
                              size_t size,
                              const Pds::TypeId& typeId,
                              const O2OXtcSrc& src) = 0 ;

  /// method called when the driver closes a group in the file
  virtual void closeContainers(hdf5pp::Group group) = 0 ;

private:

  typedef std::map<hdf5pp::Group,hdf5pp::Group> Group2Group ;

  typedef CvtDataContainer<CvtDataContFactoryDef<H5DataTypes::XtcClockTimeStamp> > XtcClockTimeCont ;

  // Data members
  const std::string m_typeGroupName ;
  const hsize_t m_chunk_size ;
  const int m_deflate ;
  std::stack<hdf5pp::Group> m_groups ;
  CvtGroupMap m_group2group ;
  XtcClockTimeCont* m_timeCont ;

};

} // namespace O2OTranslator

#endif // O2OTRANSLATOR_EVTDATATYPECVT_H
