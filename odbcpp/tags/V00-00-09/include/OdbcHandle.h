#ifndef ODBCPP_ODBCHANDLE_H
#define ODBCPP_ODBCHANDLE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcHandle.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <boost/shared_ptr.hpp>
#include <unixodbc/sql.h>

//----------------------
// Base Class Headers --
//----------------------

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
 *  C++ wrapper class for various types of SQL handles
 *
 *  This software was developed for the LUSI project.  If you use all or
 *  part of it, please give an appropriate acknowledgement.
 *
 *  @see AdditionalClass
 *
 *  @version $Id$
 *
 *  @author Andrei Salnikov
 */

namespace odbcpp {

// Helper types
struct OdbcEnv {
  typedef OdbcEnv parent_type ;
  typedef SQLHENV handle_type ;
  enum { typecode = SQL_HANDLE_ENV } ;
  static const char* type_name() { return "OdbcEnvH" ; }
};
struct OdbcConn {
  typedef OdbcEnv parent_type ;
  typedef SQLHDBC handle_type ;
  enum { typecode = SQL_HANDLE_DBC } ;
  static const char* type_name() { return "OdbcConnH" ; }
};
struct OdbcStmt {
  typedef OdbcConn parent_type ;
  typedef SQLHSTMT handle_type ;
  enum { typecode = SQL_HANDLE_STMT } ;
  static const char* type_name() { return "OdbcStmtH" ; }
};
struct OdbcDesc {
  typedef OdbcConn parent_type ;
  typedef SQLHDESC handle_type ;
  enum { typecode = SQL_HANDLE_DESC } ;
  static const char* type_name() { return "OdbcDescH" ; }
};


/// Tha handle class itself
template <typename T>
class OdbcHandle  {
public:

  typedef typename T::handle_type handle_type ;
  typedef OdbcHandle<typename T::parent_type> parent_type ;
  typedef typename parent_type::handle_type parent_handle_type ;
  enum { typecode = T::typecode } ;

  // factory method
  static OdbcHandle make( parent_type parent, bool free = true ) {
    return OdbcHandle( newPtr( parent.get(), free ) ) ;
  }

  // Default constructor
  OdbcHandle () : m_handle() {}

  // Destructor
  ~OdbcHandle () {}

  // get access to the handle
  handle_type* get() const { return m_handle.get() ; }
  handle_type* operator->() const { return m_handle.get() ; }
  handle_type& operator*() const { return *m_handle.get() ; }

  // conversion to bool
  typedef const boost::shared_ptr<handle_type> OdbcHandle::*unspecified_bool_type ;
  operator unspecified_bool_type() const { return bool(m_handle) ? &OdbcHandle::m_handle : 0 ; }
  bool operator!() const { return ! m_handle ; }

  // reset to nothing
  void reset() { m_handle.reset() ; }

protected:

  // construct from a boost pointer
  explicit OdbcHandle ( boost::shared_ptr<handle_type> p ) : m_handle(p) {}

  /// helper deleter class for boost shared_ptr
  struct OdbcHandle_Deleter  {
    OdbcHandle_Deleter( bool free ) : m_free(free) {}
    void operator() ( handle_type* handle ) {
      if ( m_free and handle ) SQLFreeHandle ( typecode, SQLHANDLE(*handle) );
      delete handle ;
    }
  private:
    bool m_free ;
  };

  static boost::shared_ptr<handle_type> newPtr ( parent_handle_type* parent, bool free ) {
    handle_type* newh = new handle_type ;
    SQLRETURN status = SQLAllocHandle ( typecode, parent ? *parent : SQL_NULL_HANDLE, newh ) ;
    if ( not SQL_SUCCEEDED(status) ) {
      delete newh ;
      newh = 0 ;
    }
    return boost::shared_ptr<handle_type> ( newh, OdbcHandle_Deleter( free ) ) ;
  }

private:

  // Data members

  boost::shared_ptr<handle_type> m_handle ;

};

template <typename T>
std::ostream&
operator<<( std::ostream& o, OdbcHandle<T> h  ) {
  std::ios_base::fmtflags oflags = o.flags() ;
  o.setf( std::ios_base::hex, std::ios_base::basefield ) ;
  o.setf( std::ios_base::showbase ) ;
  o << T::type_name() << "<" << h.get() << ">" ;
  o.flags ( oflags ) ;
  return o ;
}

} // namespace odbcpp

#endif // ODBCPP_ODBCHANDLE_H
