//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class NameIter...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/NameIter.h"

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
NameIter::NameIter (const Group& group)
  : m_group(group)
  , m_nlinks(0) 
  , m_idx(0)
{
  // get the number of links in a group
  H5G_info_t g_info;
  if (H5Gget_info(*m_group.m_id, &g_info) < 0) {
    throw Hdf5CallException( "GroupIter", "H5Gget_info") ;
  }
  m_nlinks = g_info.nlinks;
}

//--------------
// Destructor --
//--------------
NameIter::~NameIter ()
{
}

// Returns next name.
std::string 
NameIter::next()
{
  const int maxsize = 255;
  char buf[maxsize+1];

  // first try with the fixed buffer size
  ssize_t size = H5Lget_name_by_idx(*m_group.m_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, m_idx, buf, maxsize, H5P_DEFAULT);
  if (size < 0) {
    throw Hdf5CallException( "Group::name", "H5Iget_name") ;
  }
  if (size == 0) {
    // name is not known
    m_idx ++;
       return std::string();
  }
  if (size <= maxsize) {
    // name has fit into buffer
    m_idx ++;
        return buf;
  }

  // another try with dynamically allocated buffer
  char* dbuf = new char[size+1];
  H5Lget_name_by_idx(*m_group.m_id, ".", H5_INDEX_NAME, H5_ITER_NATIVE, m_idx, dbuf, size, H5P_DEFAULT);
  std::string res(dbuf);
  delete [] dbuf;
  
  m_idx ++;
  return res;
}

} // namespace hdf5pp
