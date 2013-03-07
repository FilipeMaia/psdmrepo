//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class test_compress...
//
// Author List:
//      Andy Salnikov
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------
#include <string>
#include <iostream>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <errno.h>
#include <zlib.h>
#include <sys/time.h>
#include <sys/resource.h>

//----------------------
// Base Class Headers --
//----------------------
#include "AppUtils/AppBase.h"
#include "PSTime/Time.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "AppUtils/AppCmdArg.h"
#include "AppUtils/AppCmdOpt.h"
#include "MsgLogger/MsgLogger.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------

namespace pazlib {

//
//  Application class
//
class test_compress : public AppUtils::AppBase {
public:

  // Constructor
  explicit test_compress ( const std::string& appName ) ;

  // destructor
  ~test_compress () ;

protected :

  /**
   *  Main method which runs the whole application
   */
  virtual int runApp () ;

  /// read input file
  int read();

  /// write output file
  int write();

private:

  // more command line options and arguments
  AppUtils::AppCmdOpt<int> m_levelOpt ;
  AppUtils::AppCmdOpt<int> m_repeatOpt ;
  AppUtils::AppCmdOpt<unsigned long> m_sizeOpt ;
  AppUtils::AppCmdArg<std::string> m_inputArg;
  AppUtils::AppCmdArg<std::string> m_outputArg;

  // other data members
  Bytef* m_input;
  unsigned long m_inputSize;
  Bytef* m_output;
  unsigned long m_outputSize;

};

//----------------
// Constructors --
//----------------
test_compress::test_compress ( const std::string& appName )
  : AppUtils::AppBase( appName )
  , m_levelOpt( 'c', "level", "number", "compression level, number 0-9, def: 1", 1 )
  , m_repeatOpt( 'n', "repeat", "number", "number of repeats, def: 1", 1 )
  , m_sizeOpt( 's', "size", "number", "max size to compress, def: 0", 0UL )
  , m_inputArg("input-file", "path to the input file")
  , m_outputArg("output-file", "path to the output file")
  , m_input(0)
  , m_inputSize(0)
  , m_output(0)
  , m_outputSize(0)
{
  addOption( m_levelOpt ) ;
  addOption( m_repeatOpt ) ;
  addOption( m_sizeOpt ) ;
  addArgument( m_inputArg ) ;
  addArgument( m_outputArg ) ;
}

//--------------
// Destructor --
//--------------
test_compress::~test_compress ()
{
  delete [] m_input;
  delete [] m_output;
}

/**
 *  Main method which runs the whole application
 */
int
test_compress::runApp ()
{
  int stat = read();
  if (stat != 0) return stat;

  // limit input data size
  if (m_sizeOpt.value() and m_sizeOpt.value() < m_inputSize) {
    m_inputSize = m_sizeOpt.value();
  }

  uLong adler = adler32(0L, Z_NULL, 0);
  adler = adler32(adler, m_input, m_inputSize);
  std::cout << "adler32: " << std::hex << adler << std::dec << std::endl;

  uLong adler1 = adler32(0L, Z_NULL, 0);
  uLong adler2 = adler32(0L, Z_NULL, 0);
  adler1 = adler32(adler1, m_input, m_inputSize/2);
  adler2 = adler32(adler2, m_input+m_inputSize/2, m_inputSize-m_inputSize/2);
  adler = adler32(0L, NULL, 0);
  adler = adler32_combine(adler, adler1, m_inputSize/2);
  adler = adler32_combine(adler, adler2, m_inputSize-m_inputSize/2);
  std::cout << "adler32 combined: " << std::hex << adler << std::dec << std::endl;

  // guess the size of output data
  uLong outBufSize = m_inputSize + m_inputSize/100 + 12;
  m_output = new Bytef[outBufSize];

  struct rusage ru0;
  stat = getrusage(RUSAGE_SELF, &ru0);
  PSTime::Time t0 = PSTime::Time::now();

  for (int count = m_repeatOpt.value(); count != 0 ; -- count) {

    // call compression
    m_outputSize = outBufSize;
    stat = compress2 (m_output, &m_outputSize, m_input, m_inputSize, m_levelOpt.value());
    if (stat == Z_BUF_ERROR) {
      std::cerr << "output buffer size is too small" << std::endl;
      return stat;
    } else if (stat != Z_OK) {
      std::cerr << "compress2() returned code " << stat << std::endl;
      return stat;
    }

  }

  struct rusage ru1;
  stat = getrusage(RUSAGE_SELF, &ru1);
  PSTime::Time t1 = PSTime::Time::now();
  std::cout << "user sec: " << (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) +
      (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec)/1e6 << '\n';
  std::cout << "sys  sec: " << (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) +
      (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec)/1e6 << '\n';
  std::cout << "real sec: " << (t1-t0) << '\n';

  stat = write();
  if (stat != 0) return stat;


  // return 0 on success, other values for error (like main())
  return 0 ;
}


int
test_compress::read()
{
  // open the file
  int fd = open(m_inputArg.value().c_str(), O_RDONLY);
  if (fd < 0) {
    perror("failed to open input file");
    return errno;
  }

  // get file size
  struct stat sstat;
  if (fstat(fd, &sstat) < 0) {
    perror("failed to stat input file");
    return errno;
  }
  m_inputSize = sstat.st_size;

  // allocate buffer
  m_input = new Bytef[m_inputSize];

  // read it
  ssize_t nread = ::read(fd, m_input, m_inputSize);
  if (nread != ssize_t(m_inputSize)) {
    perror("failed to read input file");
    return errno;
  }

  close(fd);

  return 0;
}

int
test_compress::write()
{
  // open the file
  int fd = open(m_outputArg.value().c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
  if (fd < 0) {
    perror("failed to open output file");
    return errno;
  }

  // read it
  ssize_t nwr = ::write(fd, m_output, m_outputSize);
  if (nwr != ssize_t(m_outputSize)) {
    perror("failed to read output file");
    return errno;
  }

  close(fd);

  return 0;
}

}


// this defines main()
APPUTILS_MAIN(pazlib::test_compress)
