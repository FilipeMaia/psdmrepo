//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Group...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "hdf5pp/Group.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Exceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  // deleter for  boost smart pointer
  struct GroupPtrDeleter {
    void operator()( hid_t* id ) {
      if ( id ) H5Gclose ( *id );
      delete id ;
    }
  };

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace hdf5pp {

//----------------
// Constructors --
//----------------
Group::Group ( hid_t grp )
  : m_id( new hid_t(grp), ::GroupPtrDeleter() )
{
}

//--------------
// Destructor --
//--------------
Group::~Group ()
{
}

// factory methods
Group
Group::createGroup ( hid_t parent, const std::string& name )
{
  MsgLog("hdf5pp::Group", debug, "Group::createGroup: parent=" << parent << " name=" << name ) ;
  // allow creation of intermediate directories
  hid_t lcpl_id = H5Pcreate( H5P_LINK_CREATE ) ;
  H5Pset_create_intermediate_group( lcpl_id, 1 ) ;
  hid_t f_id = H5Gcreate2 ( parent, name.c_str(), lcpl_id, H5P_DEFAULT, H5P_DEFAULT ) ;
  H5Pclose( lcpl_id ) ;
  if ( f_id < 0 ) {
    throw Hdf5CallException( "Group::createGroup", "H5Gcreate2") ;
  }
  return Group(f_id) ;
}

Group
Group::openGroup ( hid_t parent, const std::string& name )
{
  hid_t f_id = H5Gopen2 ( parent, name.c_str(), H5P_DEFAULT ) ;
  if ( f_id < 0 ) {
    throw Hdf5CallException( "Group::openGroup", "H5Gopen2") ;
  }
  return Group(f_id) ;
}

bool
Group::hasChild ( const std::string& name ) const
{
  std::string child = name;
  std::string::size_type p = name.find('/');
  if (p != std::string::npos) child.erase(p);
    
  // check that the group exists
  hid_t lapl_id = H5Pcreate( H5P_LINK_ACCESS ) ;
  htri_t stat = H5Lexists ( *m_id, child.c_str(), lapl_id ) ;
  H5Pclose( lapl_id ) ;
  if (p == std::string::npos) return stat>0 ;

  if (stat <= 0) return false;
  
  // open child group
  hid_t f_id = H5Gopen2 ( *m_id, child.c_str(), H5P_DEFAULT ) ;
  if ( f_id < 0 ) {
    throw Hdf5CallException( "Group::openGroup", "H5Gopen2") ;
  }
  return Group(f_id).hasChild(std::string(name, p+1));
}

// Create soft link
void
Group::makeSoftLink(const std::string& targetPath, const std::string& linkName)
{
  herr_t err = H5Lcreate_soft(targetPath.c_str(), *m_id, linkName.c_str(), H5P_DEFAULT, H5P_DEFAULT);
  if ( err < 0 ) {
    throw Hdf5CallException("Group::makeSoftLink", "H5Lcreate_soft");
  }
}

// close the group
void
Group::close()
{
  m_id.reset();
}

// get group name (absolute)
std::string 
Group::name() const
{
  const int maxsize = 255;
  char buf[maxsize+1];

  // first try with the fixed buffer size
  ssize_t size = H5Iget_name(*m_id, buf, maxsize+1);
  if (size < 0) {
    throw Hdf5CallException( "Group::name", "H5Iget_name") ;
  }
  if (size == 0) {
    // name is not known
    return std::string();
  }
  if (size <= maxsize) {
    // name has fit into buffer
    return buf;
  }

  // another try with dynamically allocated buffer
  char* dbuf = new char[size+1];
  H5Iget_name(*m_id, dbuf, size+1);
  std::string res(dbuf);
  delete [] dbuf;
  return res;
}

// get group name (relative to some parent)
std::string 
Group::basename() const
{
  const std::string& path = name();
  std::string::size_type p = path.rfind('/');
  if (p == std::string::npos) return path;
  return std::string(path, p+1);
}

// groups can be used as keys for associative containers, need compare operators
bool
Group::operator<( const Group& other ) const
{
  // if this group is None then it is less that other group that is not None
  if ( not m_id.get() ) return other.m_id.get() ;

  // this group is not None, but other is None - false
  if ( not other.m_id.get() ) return false ;

  // compare IDs
  return *m_id < *(other.m_id) ;
}

bool
Group::operator==( const Group& other ) const
{
  // if this group is None then it is less that other group that is not None
  if ( not m_id.get() ) return not other.m_id.get() ;

  // this group is not None, but other is None - false
  if ( not other.m_id.get() ) return false ;

  // compare IDs
  return *m_id == *(other.m_id) ;
}

} // namespace hdf5pp
