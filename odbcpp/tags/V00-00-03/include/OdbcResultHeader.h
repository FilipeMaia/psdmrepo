#ifndef ODBCPP_ODBCRESULTHEADER_H
#define ODBCPP_ODBCRESULTHEADER_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class OdbcResultHeader.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <vector>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "odbcpp/OdbcHandle.h"
#include "odbcpp/OdbcColumnDescr.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

/**
 *  C++ interface for ODBC result set header
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

class OdbcResultHeader  {
public:

  typedef std::vector<OdbcColumnDescr> container_type ;
  typedef container_type::const_iterator const_iterator ;
  typedef container_type::size_type size_type ;

  // Default constructor
  OdbcResultHeader ( OdbcHandle<OdbcStmt> stmtH ) ;

  // Destructor
  ~OdbcResultHeader () ;

  // get number of columns in the result
  container_type::size_type nColumns() const { return m_columns.size() ; }

  // Returns true for empty header
  bool empty() const { return m_columns.empty() ; }

  const_iterator begin() const { return m_columns.begin() ; }
  const_iterator end() const { return m_columns.end() ; }

  // get the column description for given column name, returns zero pointer for
  // non-existing column names
  const OdbcColumnDescr* findColumn( const std::string& columnName ) const ;

protected:

private:

  // data members
  container_type m_columns ;

};

} // namespace odbcpp

#endif // ODBCPP_ODBCRESULTHEADER_H
