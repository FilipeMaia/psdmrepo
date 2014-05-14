#ifndef XTCINPUT_RUNFILEITERLIST_H
#define XTCINPUT_RUNFILEITERLIST_H

//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RunFileIterList.
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <queue>
#include <map>
#include <list>

//----------------------
// Base Class Headers --
//----------------------
#include "XtcInput/RunFileIterI.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/MergeMode.h"

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
 *  @brief Implementation of RunFileIterI interface based on static list of files.
 *
 *  This implementation is useful for working with non-live data.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Andy Salnikov
 */

class RunFileIterList : public RunFileIterI {
public:

  /**
   *  @brief Make iterator instance.
   *
   *  Constructor takes sequence of XtcFileName in the form of iterators.
   *
   *  @param[in] begin     Iterator pointing to the beginning of XtcFileName sequence
   *  @param[in] end       Iterator pointing to the end of XtcFileName sequence
   *  @param[in] mergeMode Specifies how files are merged
   *  @param[in] liveTimeout Specifies timeout in second when reading live data
   */
  template <typename Iter>
  RunFileIterList (Iter begin, Iter end, MergeMode mergeMode, unsigned liveTimeout = 0)
    : RunFileIterI()
    , m_mergeMode(mergeMode)
    , m_liveTimeout(liveTimeout)
    , m_run(0)
  {
    switch (m_mergeMode) {
    case MergeFileName:
      for (; begin != end; ++ begin) m_runs[begin->run()].push_back(*begin);
      break;
    case MergeOneStream:
    case MergeNoChunking:
      for (; begin != end; ++ begin) m_runs[0].push_back(*begin);
      break;
    }
  }

  // Destructor
  virtual ~RunFileIterList () ;

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
  
  typedef std::map<unsigned, std::list<XtcFileName> > RunFiles;

  RunFiles m_runs;
  MergeMode m_mergeMode;
  unsigned m_liveTimeout;
  unsigned m_run;

};

} // namespace XtcInput

#endif // XTCINPUT_RUNFILEITERLIST_H
