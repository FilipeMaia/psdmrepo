//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class LiveFilesDB...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/LiveFilesDB.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/filesystem.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "RdbMySQL/Query.h"
#include "RdbMySQL/Result.h"
#include "RdbMySQL/Row.h"
#include "RdbMySQL/RowIter.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace fs = boost::filesystem;

namespace {

  const char* logger = "XtcInput.LiveFilesDB";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
LiveFilesDB::LiveFilesDB (const std::string& connStr, const std::string& table, const std::string& dir)
  : m_conn(connStr)
  , m_table(table)
  , m_dir(dir)
{
}

//--------------
// Destructor --
//--------------
LiveFilesDB::~LiveFilesDB ()
{
}

/**
 *  @brief Returns the list of files for given run
 *
 *  @param[in] expId    Experiment id
 *  @param[in] run      Run number
 */
std::vector<XtcFileName>
LiveFilesDB::files(unsigned expId, unsigned run)
{
  std::vector<XtcFileName> result;

  if (not m_conn.open()) {
    MsgLog(logger, error, "Failed to open database connection");
    return result;
  }

  // build query
  RdbMySQL::Query q(m_conn);

  std::string qstr = "SELECT stream, chunk, dirpath FROM ? WHERE exper_id = ?? AND run = ??";
  std::auto_ptr<RdbMySQL::Result> res(q.executePar(qstr, m_table, expId, run));

  RdbMySQL::RowIter iter(*res);
  while (iter.next()) {
    const RdbMySQL::Row& row = iter.row();

    unsigned stream = 0, chunk = 0;
    std::string dssPath;
    row.at(0, stream);
    row.at(1, chunk);
    row.at(2, dssPath);

    bool small = false; // TODO: there is currently no code in place to support getting 
                        // small data files  in live mode

    if (dssPath.empty()) {
      XtcFileName fname(m_dir, expId, run, stream, chunk, small);
      MsgLog(logger, debug, "LiveFilesDB::files - found file " << fname.path());
      result.push_back(fname);
    } else{
      // replace DSS directory name with exported directory
      fs::path path(m_dir);
      path /= fs::path(dssPath).filename();
      XtcFileName fname(path.string());
      MsgLog(logger, debug, "LiveFilesDB::files - found file " << fname.path());
      result.push_back(fname);
    }

  }
  return result;
}

} // namespace XtcInput
