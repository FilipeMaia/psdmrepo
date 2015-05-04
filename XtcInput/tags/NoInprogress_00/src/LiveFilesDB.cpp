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
LiveFilesDB::LiveFilesDB (const std::string& connStr, const std::string& table, 
                          const std::string& dir, bool small)
  : m_conn(connStr)
  , m_table(table)
  , m_dir(dir)
  , m_small(small)
{
  if (m_small) {
    fs::path dirpath(m_dir);
    dirpath /= "smalldata";
    m_dir = dirpath.string();
  }
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

  // The file table that we are reading contains records like
  // +----------+-----+--------+-------+---------------------+---------------+--------------------------------------------+
  // | exper_id | run | stream | chunk | open                | host          | dirpath                                    |
  // +----------+-----+--------+-------+---------------------+---------------+--------------------------------------------+
  // |      573 |  69 |     80 |     0 | 1425567558412694000 | ioc-fee-rec02 | /reg/d/cameras/ioc-fee-rec02/daq/xtc/e573-r0069-s80-c00.xtc      | 
  // |      575 |   1 |      0 |     0 | 1425498200940395664 | daq-cxi-dss01 | /u2/pcds/pds/cxi/e575/e575-r0001-s00-c00.xtc                     | 
  //
  // that is, the experiment id, run, stream, chunk, and a dirpath - for where the DAQ is writing the files.
  // This table only records the large xtc files - not the .smd.xtc files

  std::string qstr = "SELECT stream, chunk, dirpath FROM ? WHERE exper_id = ?? AND run = ??";
  std::auto_ptr<RdbMySQL::Result> res(q.executePar(qstr, m_table, expId, run));

  MsgLog(logger, debug, "LiveFilesDB::files - querying database for expId=" << expId << " run=" << run);

  RdbMySQL::RowIter iter(*res);
  while (iter.next()) {
    const RdbMySQL::Row& row = iter.row();

    unsigned stream = 0, chunk = 0;
    std::string dssPath;
    row.at(0, stream);
    row.at(1, chunk);
    row.at(2, dssPath);

    if (dssPath.empty()) {
      XtcFileName fname(m_dir, expId, run, stream, chunk, m_small);
      MsgLog(logger, debug, "LiveFilesDB::files - database entry found for stream=" << stream 
             << " chunk=" << chunk << ", but no dss path. Using dir & small parameters, "
             << " adding expected file from mover: " << fname.path());
      result.push_back(fname);
    } else {
      // since there is a path, re-use the basename (modifying for small if need be) but replace the
      // directory
      fs::path dbPath(dssPath);
      XtcFileName xtcBaseName(dbPath.filename().string());
      if (m_small) {
        xtcBaseName = XtcFileName(xtcBaseName.smallBasename());
      }
      fs::path path(m_dir);
      path /= fs::path(xtcBaseName.path());
      XtcFileName fname(path.string());
      MsgLog(logger, debug, "LiveFilesDB::files - database entry found for stream=" << stream 
             << " chunk=" << chunk << " with dssPath. Using dir & small parametes, "
             << " adding expcted file from mover: " << fname.path());
      result.push_back(fname);
    }

  }
  return result;
}
  
} // namespace XtcInput
