#ifndef XTCINPUT_LIVEFILESDB_H
#define XTCINPUT_LIVEFILESDB_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LiveFilesDB.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <vector>
#include <boost/utility.hpp>

//----------------------
// Base Class Headers --
//----------------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "RdbMySQL/Conn.h"
#include "XtcInput/XtcFileName.h"

//------------------------------------
// Collaborating Class Declarations --
//------------------------------------

//		---------------------
// 		-- Class Interface --
//		---------------------

namespace XtcInput {

/// @addtogroup XtcInput

/**
 *  @ingroup XtcInput
 *
 *  @brief Class which implements interface to migration database
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class LiveFilesDB : boost::noncopyable {
public:

  /**
   *  @brief Make an instance
   *
   *  @param[in] connStr  Connection string.
   *  @param[in] table    Table name
   *  @param[in] dir      Directory to look for live files
   */
  LiveFilesDB(const std::string& connStr, const std::string& table, const std::string& dir);

  // Destructor
  ~LiveFilesDB () ;

  /**
   *  @brief Returns the list of files for given run
   *
   *  @param[in] expId    Experiment id
   *  @param[in] run      Run number
   */
  std::vector<XtcFileName> files(unsigned expId, unsigned run);

protected:

private:

  RdbMySQL::Conn m_conn;      ///< Connection to mysql database
  const std::string m_table;  ///< Name of the table containing file list
  const std::string m_dir;    ///< Directory to look for live files

};

} // namespace XtcInput

#endif // XTCINPUT_LIVEFILESDB_H
