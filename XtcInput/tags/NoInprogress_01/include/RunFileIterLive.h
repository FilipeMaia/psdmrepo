#ifndef XTCINPUT_RUNFILEITERLIVE_H
#define XTCINPUT_RUNFILEITERLIVE_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RunFileIterLive.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <set>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

//----------------------
// Base Class Headers --
//----------------------
#include "XtcInput/RunFileIterI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/LiveFilesDB.h"
#include "IData/Dataset.h"

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
 *  @brief Implementation of RunFileIterI interface working with live data.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class RunFileIterLive : public RunFileIterI {
public:

  /**
   *  @brief Make iterator instance.
   *
   *  Constructor takes sequence of run numbers in the form of iterators.
   *
   *  @param[in] begin     Iterator pointing to the beginning of run number sequence
   *  @param[in] end       Iterator pointing to the end of run number sequence
   *  @param[in] expNum    Experiment number
   *  @param[in] stream    Stream number, or -1 for all stream, -2 for any one stream, -3 for ranges of streams
   *  @param[in] ds_streams Arnges of streams
   *  @param[in] liveTimeout Specifies timeout in second when reading live data
   *  @param[in] runLiveTimeout Specifies timeout in second when waiting for a new run to show in the database when reading live data
   *  @param[in] dbConnStr Database connection string
   *  @param[in] table     Database table name
   *  @param[in] dir       Directory to look for live files
   *  @param[in] small     look for small data files
   */
  template <typename Iter>
    RunFileIterLive (Iter begin, Iter end, unsigned expNum, int stream, const IData::Dataset::Streams& ds_streams, unsigned liveTimeout, unsigned runLiveTimeout,
                     const std::string& dbConnStr, const std::string& table, const std::string& dir, bool small)
    : RunFileIterI()
    , m_runs(begin, end)
    , m_expNum(expNum)
    , m_stream(stream)
    , m_ds_streams(ds_streams)
    , m_liveTimeout(liveTimeout)
    , m_runLiveTimeout(runLiveTimeout)
    , m_run(0)
    , m_filesdb(boost::make_shared<LiveFilesDB>(dbConnStr, table, dir, small))
  {
  }

  // Destructor
  virtual ~RunFileIterLive () ;

  /**
   *  @brief Return stream iterator for next run.
   *
   *  Zero pointer is returned after last run.
   */
  virtual boost::shared_ptr<StreamFileIterI> next();

  /**
   *  @brief Return run number for the set of files returned from last next() call.
   */
  virtual unsigned run() const;

protected:

private:

  typedef std::set<unsigned> Runs;
  
  Runs m_runs;
  unsigned m_expNum;
  int m_stream;
  IData::Dataset::Streams m_ds_streams;
  unsigned m_liveTimeout;
  unsigned m_runLiveTimeout;
  unsigned m_run;
  boost::shared_ptr<LiveFilesDB> m_filesdb;

};

} // namespace XtcInput

#endif // XTCINPUT_RUNFILEITERLIVE_H
