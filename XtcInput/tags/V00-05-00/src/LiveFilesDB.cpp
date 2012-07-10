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
#include "MsgLogger/MsgLogger.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "RdbMySQL/Query.h"
#include "RdbMySQL/Result.h"
#include "RdbMySQL/Row.h"
#include "RdbMySQL/RowIter.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace {

  const char* logger = "LiveFilesDB";

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//----------------
// Constructors --
//----------------
LiveFilesDB::LiveFilesDB (const std::string& connStr, const std::string& table,
    const std::string& dssDir, const std::string& anaDir)
  : m_conn(connStr)
  , m_table(table)
  , m_dssDir(dssDir)
  , m_anaDir(anaDir)
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
    std::string dirpath;
    row.at(0, stream);
    row.at(1, chunk);
    row.at(2, dirpath);

    if (dirpath.empty()) {
      XtcFileName fname("", expId, run, stream, chunk);
      MsgLog(logger, debug, "LiveFilesDB::files - found file " << fname.path());
      result.push_back(fname);
    } else{
      // replace DSS directory name with exported directory
      if (dirpath.compare(0, m_dssDir.size(), m_dssDir) == 0) {
        dirpath.replace(0, m_dssDir.size(), m_anaDir);
      }
      XtcFileName fname(dirpath);
      MsgLog(logger, debug, "LiveFilesDB::files - found file " << fname.path());
      result.push_back(fname);
    }

  }
  return result;
}

} // namespace XtcInput
