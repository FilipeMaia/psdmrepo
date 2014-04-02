//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class XtcFilterTest...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>

//----------------------
// Base Class Headers --
//----------------------
#include "AppUtils/AppBase.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdArg.h"
#include "AppUtils/AppCmdOptList.h"
#include "AppUtils/AppCmdOptToggle.h"
#include "MsgLogger/MsgLogger.h"
#include "XtcInput/XtcFilter.h"
#include "XtcInput/XtcFilterTypeId.h"
#include "XtcInput/XtcChunkDgIter.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace XtcInput {

//
//  Application class
//
class XtcFilterTest : public AppUtils::AppBase {
public:

  // Constructor
  explicit XtcFilterTest ( const std::string& appName ) ;

  // destructor
  ~XtcFilterTest () ;

protected :

  /**
   *  Main method which runs the whole application
   */
  virtual int runApp () ;

private:

  // more command line options and arguments
  AppUtils::AppCmdOptList<int> m_keepOpt ;
  AppUtils::AppCmdOptList<int> m_discardOpt ;
  AppUtils::AppCmdOptToggle m_keepContOpt ;
  AppUtils::AppCmdOptToggle m_keepDgramOpt ;
  AppUtils::AppCmdOptToggle m_keepAnyOpt ;
  AppUtils::AppCmdArg<std::string> m_inputArg ;
  AppUtils::AppCmdArg<std::string> m_outputArg ;

};

//----------------
// Constructors --
//----------------
XtcFilterTest::XtcFilterTest ( const std::string& appName )
  : AppUtils::AppBase( appName )
  , m_keepOpt( parser(), "k,keep", "number", "TypeId numbers to keep" )
  , m_discardOpt( parser(), "d,discard", "number", "TypeId numbers to discard" )
  , m_keepContOpt( parser(), "c,keep-empty-cont", "keep empty XTC containers", false )
  , m_keepDgramOpt( parser(), "g,keep-empty-dgram", "keep empty datagrams", false )
  , m_keepAnyOpt( parser(), "a,keep-any", "keep Any XTC", false )
  , m_inputArg( parser(), "xtc-input", "input file name" )
  , m_outputArg( parser(), "xtc-output", "output file name" )
{
}

//--------------
// Destructor --
//--------------
XtcFilterTest::~XtcFilterTest ()
{
}

/**
 *  Main method which runs the whole application
 */
int
XtcFilterTest::runApp ()
{
  // build keep/discard lists
  XtcFilterTypeId::IdList keep;
  for (AppUtils::AppCmdOptList<int>::const_iterator it = m_keepOpt.begin(); it != m_keepOpt.end(); ++ it) {
    keep.push_back(Pds::TypeId::Type(*it));
  }
  XtcFilterTypeId::IdList discard;
  for (AppUtils::AppCmdOptList<int>::const_iterator it = m_discardOpt.begin(); it != m_discardOpt.end(); ++ it) {
    discard.push_back(Pds::TypeId::Type(*it));
  }
  if (not keep.empty()) {
    WithMsgLogRoot(info, str) {
      str << "Keep list:";
      for (XtcFilterTypeId::IdList::const_iterator it = keep.begin(); it != keep.end(); ++ it) {
        str << " " << Pds::TypeId::name(*it);
      }
    }
  }
  if (not discard.empty()) {
    WithMsgLogRoot(info, str) {
      str << "Discard list:";
      for (XtcFilterTypeId::IdList::const_iterator it = discard.begin(); it != discard.end(); ++ it) {
        str << " " << Pds::TypeId::name(*it);
      }
    }
  }

  // open output file
  int fd = creat(m_outputArg.value().c_str(), 0666);
  if (fd < 0) {
    perror("XtcFilterTest");
    return errno;
  }

  // instantiate filter
  XtcFilter<XtcFilterTypeId> filter(XtcFilterTypeId(keep, discard), m_keepContOpt.value(), m_keepDgramOpt.value(), m_keepAnyOpt.value());

  const int DgSize = 16*1024*1024;
  XtcChunkDgIter dgIter(XtcFileName(m_inputArg.value()), DgSize);

  boost::shared_ptr<DgHeader> hptr = dgIter.next();
  char* buffer = new char[DgSize];
  while (hptr) {

    Dgram::ptr dg = hptr->dgram();

    size_t size = filter.filter(dg.get(), buffer);

    MsgLogRoot(trace, "filter: orig size = " << (sizeof(Pds::Dgram) + dg->xtc.sizeofPayload()) << " final size = " << size);

    if (size) {
      write(fd, buffer, size);
    }

    hptr = dgIter.next();
  }

  close(fd);

  return 0 ;
}

} // namespace XtcInput


// this defines main()
APPUTILS_MAIN(XtcInput::XtcFilterTest)
