//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class RunFileIterList...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "XtcInput/RunFileIterList.h"

//-----------------
// C/C++ Headers --
//-----------------
#include <boost/make_shared.hpp>

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "XtcInput/StreamFileIterList.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//--------------
// Destructor --
//--------------
RunFileIterList::~RunFileIterList ()
{
}

/**
 *  @brief Return stream iterator for next run.
 *  
 *  Zero pointer is returned after last run.
 */
boost::shared_ptr<StreamFileIterI> 
RunFileIterList::next()
{
  boost::shared_ptr<StreamFileIterI> next;
  if (not m_runs.empty()) {
    RunFiles::iterator s = m_runs.begin();
    m_run = s->first;
    next = boost::make_shared<StreamFileIterList>(s->second.begin(), s->second.end(), m_mergeMode, m_liveTimeout);
    m_runs.erase(s);
  }
  
  return next;  
}

// Return run number for the set of files returned from last next() call.
unsigned 
RunFileIterList::run() const
{
  return m_run;
}


} // namespace XtcInput
