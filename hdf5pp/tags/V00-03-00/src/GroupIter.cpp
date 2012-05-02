//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class GroupIter...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/GroupIter.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Exceptions.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace hdf5pp {

//----------------
// Constructors --
//----------------
GroupIter::GroupIter (const Group& group)
  : m_group(group)
  , m_nlinks(0) 
  , m_idx(0)
{
  // get the number of links in a group
  H5G_info_t g_info;
  if (H5Gget_info(*m_group.m_id, &g_info) < 0) {
    throw Hdf5CallException( ERR_LOC, "H5Gget_info") ;
  }
  m_nlinks = g_info.nlinks;
}

//--------------
// Destructor --
//--------------
GroupIter::~GroupIter ()
{
}

/// Returns next group
Group 
GroupIter::next()
{
  Group grp;  
  for (; not grp.valid() and m_idx < m_nlinks; ++ m_idx) {
    
    // open object
    hid_t hid = H5Oopen_by_idx(*m_group.m_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, m_idx, H5P_DEFAULT);
    if (hid < 0) {
      throw Hdf5CallException( ERR_LOC, "H5Oopen_by_idx") ;
    }

    // only care about groups
    if (H5Iget_type(hid) != H5I_GROUP) {
      H5Oclose(hid);
      continue;
    }

    // make a group object and return
    grp = Group(hid);
  }
  
  // Done iterating
  return grp;
}

} // namespace hdf5pp
