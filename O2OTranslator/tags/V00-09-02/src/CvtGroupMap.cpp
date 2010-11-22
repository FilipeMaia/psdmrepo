//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class CvtGroupMap...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------
#include "SITConfig/SITConfig.h"

//-----------------------
// This Class's Header --
//-----------------------
#include "O2OTranslator/CvtGroupMap.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace O2OTranslator {

// comparison operator for Src objects
bool
CvtGroupMap::_SrcCmp::operator()( const Pds::Src& lhs, const Pds::Src& rhs ) const
{
  if ( lhs.log() < rhs.log() ) return true ;
  if ( lhs.log() > rhs.log() ) return false ;
  if ( lhs.phy() < rhs.phy() ) return true ;
  return false ;
}


//----------------
// Constructors --
//----------------
CvtGroupMap::CvtGroupMap ()
  : m_group2group()
{
}

//--------------
// Destructor --
//--------------
CvtGroupMap::~CvtGroupMap ()
{
}

/// find a group for a given top group and Src, return invalid
/// object if not found
hdf5pp::Group
CvtGroupMap::find( hdf5pp::Group top, const Pds::Src& src ) const
{
  Group2Group::const_iterator git = m_group2group.find( top ) ;
  if ( git == m_group2group.end() ) return hdf5pp::Group() ;

  Src2Group::const_iterator dit = git->second.find( src ) ;
  if ( dit == git->second.end() ) return hdf5pp::Group() ;

  return dit->second ;
}

/// insert new mapping
void
CvtGroupMap::insert ( hdf5pp::Group top, const Pds::Src& src, hdf5pp::Group group )
{
  m_group2group[top].insert ( Src2Group::value_type(src,group) ) ;
}

/// remove all mappings for given top group
void
CvtGroupMap::erase ( hdf5pp::Group top )
{
  m_group2group.erase(top) ;
}

/// get the set of subgroups for a given top group
CvtGroupMap::GroupList
CvtGroupMap::groups( hdf5pp::Group top ) const
{
  GroupList result ;

  Group2Group::const_iterator git = m_group2group.find( top ) ;
  if ( git == m_group2group.end() ) return result ;

  for ( Src2Group::const_iterator it = git->second.begin() ; it != git->second.end() ; ++ it ) {
    result.push_back( it->second ) ;
  }

  return result ;
}


} // namespace O2OTranslator
