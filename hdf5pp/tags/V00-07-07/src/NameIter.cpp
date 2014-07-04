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
NameIter::NameIter (const Group& group, LinkType type)
  : m_group(group)
  , m_type(type)
  , m_nlinks(0) 
  , m_idx(0)
{
  // get the number of links in a group
  H5G_info_t g_info;
  if (H5Gget_info(m_group.id(), &g_info) < 0) {
    throw Hdf5CallException( ERR_LOC, "H5Gget_info") ;
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

  std::string result;
  for (; result.empty() and m_idx < m_nlinks; ++ m_idx) {

    if (m_type != Any) {
      // test for link type
      H5L_info_t linfo;
      herr_t err = H5Lget_info_by_idx(m_group.id(), ".", H5_INDEX_NAME, H5_ITER_NATIVE, m_idx, &linfo, H5P_DEFAULT);
      if (err < 0) {
        throw Hdf5CallException( ERR_LOC, "H5Lget_info_by_idx") ;
      }
      if (not (linfo.type == H5L_TYPE_SOFT and int(m_type) & int(SoftLink)) and
          not (linfo.type == H5L_TYPE_HARD and int(m_type) & int(HardLink))) continue;
    }

    // first try with the fixed buffer size
    ssize_t size = H5Lget_name_by_idx(m_group.id(), ".", H5_INDEX_NAME, H5_ITER_NATIVE, m_idx, buf, maxsize, H5P_DEFAULT);
    if (size < 0) {
      throw Hdf5CallException( ERR_LOC, "H5Lget_name_by_idx") ;
    }
    if (size == 0) {
      // name is not known
      continue;
    }
    if (size <= maxsize) {
      // name has fit into buffer
      result = buf;
    } else {
      // another try with dynamically allocated buffer
      char* dbuf = new char[size+1];
      H5Lget_name_by_idx(m_group.id(), ".", H5_INDEX_NAME, H5_ITER_NATIVE, m_idx, dbuf, size, H5P_DEFAULT);
      result = dbuf;
      delete [] dbuf;
    }

  }

  return result;
}

} // namespace hdf5pp
