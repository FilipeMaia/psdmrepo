//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class DgramReader...
//
// Author List:
//      Andrei Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/DgramReader.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <algorithm>
#include <iterator>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/format.hpp>
#include <boost/make_shared.hpp>
#include <boost/thread/thread.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "IData/Dataset.h"
#include "MsgLogger/MsgLogger.h"
#include "XtcInput/Exceptions.h"
#include "XtcInput/DgramQueue.h"
#include "XtcInput/RunFileIterList.h"
#include "XtcInput/RunFileIterLive.h"
#include "XtcInput/XtcFileName.h"
#include "XtcInput/XtcMergeIterator.h"
#include "pdsdata/xtc/Dgram.hh"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

namespace fs = boost::filesystem;

namespace {

  const char* logger = "DgramReader";

  // get directory name for the dataset
  std::string dsDirPath(const IData::Dataset& ds);

  // Find files on dist corresponding to a dataset
  void findDeadFiles(const IData::Dataset& ds, std::vector<XtcInput::XtcFileName>& files);

}

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//--------------
// Destructor --
//--------------
DgramReader::~DgramReader ()
{
}

// this is the "run" method used by the Boost.thread
void
DgramReader::operator() ()
try {

  // In non-live mode input can be a mixture of files and datasets,
  // none of datasets can specify live mode.
  // In live mode input can include datasets only, first dataset must
  // specify live mode, other datasets do not matter.

  enum {Unknown, Live, Dead} liveMode = Unknown;

  std::vector<XtcFileName> files;  // file names for "dead" mode
  IData::Dataset::Runs runs;  // run numbers for live mode
  unsigned expId = 0;
  std::string liveDir;

  // guess whether we have datasets or pure file names (or mixture)
  for (FileList::const_iterator it = m_files.begin(); it != m_files.end(); ++ it) {
    if (it->find('=') == std::string::npos) {

      // must be file name
      if (liveMode == Live) throw DatasetSpecError(ERR_LOC, "cannot specify file names in live mode");
      if (liveMode == Unknown) liveMode = Dead;
      files.push_back(XtcFileName(*it));

    } else {

      IData::Dataset ds(*it);
      bool live = liveMode == Live or ds.exists("live");
      if (live) {

        // check or set live mode
        if (liveMode == Dead) throw DatasetSpecError(ERR_LOC, "cannot mix live and non-live data");
        if (liveMode == Unknown) {
          liveMode = Live;
          // remember experiment ID as well
          expId = ds.expID();
        }

        // get directory name where to look for files
        if (liveDir.empty()) {
          liveDir = ::dsDirPath(ds);
        } else {
          std::string dir = ds.value("dir");
          if (not dir.empty() and liveDir != dir) {
            throw LiveDirError(ERR_LOC);
          }
        }

        // copy run ranges
        const IData::Dataset::Runs& dsruns = ds.runs();
        std::copy(dsruns.begin(), dsruns.end(), std::back_inserter(runs));

      } else {

        if (liveMode == Unknown) liveMode = Dead;
        // Find files on disk and add to the list
        ::findDeadFiles(ds, files);

      }

    }
  }

  // make instance of file iterator
  boost::shared_ptr<RunFileIterI> fileIter;
  if (liveMode == Dead) {

    if (not files.empty()) {
      fileIter = boost::make_shared<RunFileIterList>(files.begin(), files.end(), m_mode);
    }

  } else {

    // make a list of run numbers
    std::vector<unsigned> numbers;
    for (IData::Dataset::Runs::const_iterator ritr = runs.begin(); ritr != runs.end(); ++ ritr) {
      for (unsigned run = ritr->first; run <= ritr->second; ++ run) {
        numbers.push_back(run);
      }
    }
    if (not numbers.empty()) {
      // use default table name if none was given
      if (m_liveDbConn.empty()) m_liveDbConn = "Server=psdb.slac.stanford.edu;Database=regdb;Uid=regdb_reader";
      fileIter = boost::make_shared<RunFileIterLive>(numbers.begin(), numbers.end(), expId,
          m_liveTimeout, m_liveDbConn, m_liveTable, liveDir);
    }
  }

  if (fileIter) {

    XtcMergeIterator iter(fileIter, m_l1OffsetSec);
    Dgram dg;
    while ( not boost::this_thread::interruption_requested() ) {

      dg = iter.next();

      // stop if no datagram
      if (dg.empty()) break;

      // move it to the queue
      m_queue.push ( dg ) ;

    }

  } else {

    MsgLog(logger, warning, "no input data specified");

  }

  // tell all we are done
  m_queue.push ( Dgram() ) ;

} catch (const boost::thread_interrupted& ex) {

  // we just stop happily, remove all current datagrams from a queue
  // to make sure there is enough free spave and add end-of-data datagram just in
  // case someone needs it
  m_queue.clear();
  m_queue.push ( Dgram() ) ;

} catch ( std::exception& e ) {

  MsgLog(logger, error, "exception caught while reading datagram: " << e.what());
  // TODO: there is no way yet to stop gracefully, will just abort
  throw;

}


} // namespace XtcInput

namespace {

// get directory name for the dataset
std::string dsDirPath(const IData::Dataset& ds)
{
  // get directory name where to look for files
  std::string dir = ds.value("dir");
  if (dir.empty()) {
    boost::format fmt("/reg/d/psdm/%1%/%2%/xtc");
    fmt % ds.instrument() % ds.experiment();
    dir = fmt.str();
  }
  return dir;
}

// Find files on dist corresponding to a dataset
void findDeadFiles(const IData::Dataset& ds, std::vector<XtcInput::XtcFileName>& files)
{
  // get directory name where to look for files
  fs::path dir = ::dsDirPath(ds);
  if (not fs::is_directory(dir)) {
    throw XtcInput::DatasetDirError(ERR_LOC, dir.string());
  }

  // scan all files in directory, find matching ones
  std::map<unsigned, unsigned> filesPerRun;
  for (fs::directory_iterator fiter(dir); fiter != fs::directory_iterator(); ++ fiter) {

    if (fiter->status().type() != fs::regular_file) continue;
    
    const fs::path& path = fiter->path();
    const fs::path& basename = path.filename();
//    MsgLog(logger, debug, "matching file: " << basename);
    
    const IData::Dataset::Runs& runs = ds.runs();
    for (IData::Dataset::Runs::const_iterator ritr = runs.begin(); ritr != runs.end(); ++ ritr) {
      for (unsigned run = ritr->first; run <= ritr->second; ++ run) {
        
        // make file name regex 
        boost::format pattern("e%1%-r0*%2%-s[0-9]+-c[0-9]+[.]xtc");
        pattern % ds.expID() % run;
        boost::regex re(pattern.str());
//        MsgLog(logger, debug, "pattern: " << pattern.str());

        if (boost::regex_match(basename.string(), re)) {
          MsgLog(logger, debug, "found matching file: " << path);
          files.push_back(XtcInput::XtcFileName(path.string()));
          ++ filesPerRun[run];
        }
      }
      
    }
  }

  // Check file count per run
  const IData::Dataset::Runs& runs = ds.runs();
  for (IData::Dataset::Runs::const_iterator ritr = runs.begin(); ritr != runs.end(); ++ ritr) {
    // only check runs specified explicitly, not ranges
    if (ritr->first == ritr->second) {
      if (filesPerRun[ritr->first] == 0) {
        MsgLog(logger, warning, "no input files found for run #" << ritr->first);
      }
    }
  }

}

}
