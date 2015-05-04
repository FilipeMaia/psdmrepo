//--------------------------------------------------------------------------
// File and Version Information:
//     $Id$
//
// Description:
//     Class DgramReader...
//
// Author List:
//     Andrei Salnikov
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

  const char* logger = "XtcInput.DgramReader";

}

//             ----------------------------------------
//             -- Public Function Member Definitions --
//             ----------------------------------------

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
  const int AllStreams=-1;
  const int AnyOneStream=-2;
  const int RangeOfStreams=-3;
  int  stream = AllStreams;

  std::vector<XtcFileName> files;  // file names for "dead" mode
  IData::Dataset::Runs runs;  // run numbers for live mode
  IData::Dataset::Streams streams;  // stream ranges for live mode
  unsigned expId = 0;
  std::string liveDir;
  bool smallData = false;

  // guess whether we have datasets or pure file names (or mixture)
  for (FileList::const_iterator it = m_files.begin(); it != m_files.end(); ++ it) {
    
    IData::Dataset ds(*it);

    if (ds.exists("smd")) {
      smallData = true;
    } else {
      if (smallData == true) {
        MsgLog(logger, warning, "smalldata has been identified earlier in files argument, "
               << "but it is not present with this arg - will assume small data for live mode");
      }
    }
    // make sure the stream number and range filters are not used together
    if (ds.exists("one-stream") && ds.exists("stream")) throw DatasetSpecError(ERR_LOC, "cannot specify both 'one-stream' and 'stream'");

    if (ds.isFile()) {

      // must be file name
      if (liveMode == Live) throw DatasetSpecError(ERR_LOC, "cannot specify file names in live mode");
      if (liveMode == Unknown) liveMode = Dead;
      files.push_back(XtcFileName(*it));

    } else {

      // get stream number
      if (ds.exists("one-stream")) {
        std::string strval = ds.value("one-stream");
        if (strval.empty()) {
          // no value means AnyOneStream, check it does not conflict
          if (stream == AllStreams) {
            stream = AnyOneStream; 
          } else if (stream != AnyOneStream) {
            throw LiveStreamError(ERR_LOC);
          }
        } else {
          // string must be non-negative number
          unsigned val;
          try {
            val = boost::lexical_cast<unsigned>(strval);
          } catch (const boost::bad_lexical_cast& ex) {
            throw LiveStreamError(ERR_LOC);
          }
          // check it does not conflict
          if (stream == AllStreams) {
            stream = val; 
          } else if (stream != int(val)) {
            throw LiveStreamError(ERR_LOC);
          }
        }
      }

      // check if using ranges of streams
      if (ds.exists("stream")) {
        std::string strval = ds.value("stream");
        if (strval.empty()) {
          stream = AllStreams;       // no value means all streams
        } else {
          stream = RangeOfStreams;
          // copy stream ranges
          const IData::Dataset::Streams& dsstreams = ds.streams();
          std::copy(dsstreams.begin(), dsstreams.end(), std::back_inserter(streams));

        }
      }

      bool live = (liveMode == Live or ds.exists("live"));
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
          liveDir = ds.dirName();
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
        const IData::Dataset::NameList& strfiles = ds.files();
        if (strfiles.empty()) throw NoFilesInDataset(ERR_LOC, *it);
        for (IData::Dataset::NameList::const_iterator it = strfiles.begin(); it != strfiles.end(); ++ it) {
          files.push_back(XtcFileName(*it));
        }

      }

    }
  }

  // make instance of run file iterator
  boost::shared_ptr<RunFileIterI> runFileIter;
  if (liveMode == Dead) {

    // if one-stream is active filter file names
    if (stream != AllStreams) {
      
      // sort all files according run and stream number
      typedef std::map<unsigned, std::vector<XtcFileName> > StreamMap;
      typedef std::map<unsigned, StreamMap> RunStreamMap;
      RunStreamMap rsMap;
      for (std::vector<XtcFileName>::const_iterator it = files.begin(); it != files.end(); ++ it) {
        rsMap[it->run()][it->stream()].push_back(*it);
      }

      // clear the list for now
      files.clear();

      // copy only streams that match
      for (RunStreamMap::iterator r_it = rsMap.begin(); r_it != rsMap.end(); ++ r_it) {
        
        StreamMap& streamMap = r_it->second;
        if (stream == RangeOfStreams) {

          // filter on streams which fall into any range in the dataset specification
          for (StreamMap::const_iterator it = streamMap.begin(); it != streamMap.end(); ++ it) {
            bool evict = true;
            for (IData::Dataset::Streams::const_iterator s_it = streams.begin(); s_it != streams.end(); ++ s_it) {
              if (s_it->first <= it->first && it->first <= s_it->second) {
                evict = false;
                break;
              }
            }
            if (evict) {
              MsgLog(logger, debug, "Stream filtering mode, evict stream number: " << it->first);
            } else {
              files.insert(files.end(), it->second.begin(), it->second.end());
              MsgLog(logger, debug, "Stream filtering mode, accept stream number: " << it->first);
            }
          }

        } else {

          StreamMap::iterator it;
          if (stream == AnyOneStream) {
            // leave one stream only, try to randomize but in a reproducible way
            unsigned run = r_it->first;
            unsigned idx = run % streamMap.size();
            it = streamMap.begin();
            std::advance(it, idx);
          } else {
            // find one specific stream
            it = streamMap.find(stream);
          }
        
          if (it == streamMap.end()) {
            MsgLog(logger, debug, "one-stream filtering mode, no matching streams");
          } else {
            files.insert(files.end(), it->second.begin(), it->second.end());
            MsgLog(logger, debug, "one-stream filtering mode, stream number: " << it->first);
          }
        }
      }
    }
    
    if (not files.empty()) {
      runFileIter = boost::make_shared<RunFileIterList>(files.begin(), files.end(), m_mode);
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
      // we would prefer to use boost::make_shared below, but in C++ 03 it stops 
      // after 9 arguments (C++ 11 variadic templates would resolve this)
      boost::shared_ptr<RunFileIterLive> temp( new RunFileIterLive(numbers.begin(), numbers.end(), expId, stream, streams,
                                                                   m_liveTimeout, m_runLiveTimeout, m_liveDbConn, 
                                                                   m_liveTable, liveDir, smallData));
      runFileIter = temp;
    }
  }
  
  if (runFileIter) {

    XtcMergeIterator iter(runFileIter, m_l1OffsetSec, m_firstControlStream, 
                          m_maxStreamClockDiffSec, m_thirdEvent);
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

  // push exception message to a queue which will cause exception in consumer thread
  m_queue.push_exception(e.what());

}


} // namespace XtcInput

