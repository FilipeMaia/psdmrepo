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
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "hdf5pp/Exceptions.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char logger[] = "hdf5pp.Group";

  // deleter for  boost smart pointer
  struct GroupPtrDeleter {
    void operator()( hid_t* id ) {
      if ( id ) {
        MsgLog(logger, debug, "GroupPtrDeleter: group=" << *id) ;
        H5Gclose ( *id );
      }
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
  , m_dsCache(boost::make_shared<DsCache>())
{
  MsgLog(logger, debug, "Group ctor: " << *this) ;
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
  MsgLog(logger, debug, "Group::createGroup: parent=" << parent << " name=" << name ) ;
  // allow creation of intermediate directories
  hid_t lcpl_id = H5Pcreate( H5P_LINK_CREATE ) ;
  H5Pset_create_intermediate_group( lcpl_id, 1 ) ;
  hid_t f_id = H5Gcreate2 ( parent, name.c_str(), lcpl_id, H5P_DEFAULT, H5P_DEFAULT ) ;
  H5Pclose( lcpl_id ) ;
  if ( f_id < 0 ) {
    throw Hdf5CallException( ERR_LOC, "H5Gcreate2") ;
  }
  return Group(f_id) ;
}

Group
Group::openGroup ( hid_t parent, const std::string& name )
{
  MsgLog(logger, debug, "Group::openGroup: parent=" << parent << " name=" << name ) ;
  hid_t f_id = H5Gopen2 ( parent, name.c_str(), H5P_DEFAULT ) ;
  if ( f_id < 0 ) {
    throw Hdf5CallException( ERR_LOC, "H5Gopen2") ;
  }
  return Group(f_id) ;
}

bool
Group::hasChild ( const std::string& name ) const
{
  MsgLog(logger, debug, "Group::hasChild: this=" << *this << " name=" << name) ;

  std::string child = name;
  std::string::size_type p = name.find('/');
  if (p != std::string::npos) {
    child.erase(p);
  } else {
    // small optimization, if this is the last item in the path check cached datasets first
    DsCache::const_iterator it = m_dsCache->find(child);
    if (it != m_dsCache->end()) return true;
  }

  // check that the link exists
  hid_t lapl_id = H5Pcreate( H5P_LINK_ACCESS ) ;
  htri_t stat = H5Lexists ( *m_id, child.c_str(), lapl_id ) ;
  H5Pclose( lapl_id ) ;
  if (p == std::string::npos) return stat>0 ;

  if (stat <= 0) return false;
  
  // open child group
  hid_t f_id = H5Gopen2 ( *m_id, child.c_str(), H5P_DEFAULT ) ;
  if ( f_id < 0 ) {
    throw Hdf5CallException( ERR_LOC, "H5Gopen2") ;
  }
  return Group(f_id).hasChild(std::string(name, p+1));
}

// get parent for this group, returns non-valid object if no parent group exists
Group
Group::parent() const
{
  MsgLog(logger, debug, "Group::parent: this=" << *this);

  std::string path = name();
  if (path.empty() or path == "/") return Group();

  std::string::size_type p = path.rfind('/');
  if (p != std::string::npos) path.erase(p);

  return openGroup(*m_id, path);
}

// open existing data set
DataSet
Group::openDataSet (const std::string& name, const PListDataSetAccess& plistDSaccess) const
{
  DsCache::const_iterator it = m_dsCache->find(name);
  if (it != m_dsCache->end()) return it->second;

  DataSet res = DataSet::openDataSet ( *m_id, name, plistDSaccess ) ;
  m_dsCache->insert(std::make_pair(name, res));

  return res;
}

// Create soft link
void
Group::makeSoftLink(const std::string& targetPath, const std::string& linkName)
{
  herr_t err = H5Lcreate_soft(targetPath.c_str(), *m_id, linkName.c_str(), H5P_DEFAULT, H5P_DEFAULT);
  if ( err < 0 ) {
    throw Hdf5CallException( ERR_LOC, "H5Lcreate_soft") ;
  }
}

// Get link type.
H5L_type_t
Group::getLinkType(const std::string& linkName) const
{
  H5L_info_t linkInfo;
  herr_t err = H5Lget_info(*m_id, linkName.c_str(), &linkInfo, H5P_DEFAULT);
  if ( err < 0 ) {
    throw Hdf5CallException( ERR_LOC, "H5Lget_info") ;
  }
  return linkInfo.type;
}

// Get soft link value.
std::string
Group::getSoftLink(const std::string& linkName) const
{
  // check link type and get its size
  H5L_info_t linkInfo;
  herr_t err = H5Lget_info(*m_id, linkName.c_str(), &linkInfo, H5P_DEFAULT);
  if ( err < 0 ) {
    throw Hdf5CallException( ERR_LOC, "H5Lget_info") ;
  }
  if (linkInfo.type != H5L_TYPE_SOFT) {
    throw Hdf5CallException( ERR_LOC, "H5Lget_val") ;
  }

  // allocate memory for link value
  char buf[64];
  char* p = buf;
  if (linkInfo.u.val_size > sizeof buf) p = new char[linkInfo.u.val_size];

  // get the value
  err = H5Lget_val(*m_id, linkName.c_str(), p, linkInfo.u.val_size, H5P_DEFAULT);
  if ( err < 0 ) {
    throw Hdf5CallException( ERR_LOC, "H5Lget_val") ;
  }
  std::string res(p);

  if (p != buf) delete [] p;

  return res;
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
    throw Hdf5CallException( ERR_LOC, "H5Iget_name") ;
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

// Insertion operator dumps name and ID of the group.
std::ostream&
operator<<(std::ostream& out, const Group& grp)
{
  if (grp.valid()) {
    return out << "Group(id=" << grp.id() << ", name='" << grp.name() << "')";
  } else {
    return out << "Group(id=None)";
  }
}

} // namespace hdf5pp
