//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class test_decompress...
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
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <zlib.h>
#include <sys/time.h>
#include <sys/resource.h>

//----------------------
// Base Class Headers --
//----------------------
#include "AppUtils/AppBase.h"

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
class test_decompress : public AppUtils::AppBase {
public:

  // Constructor
  explicit test_decompress ( const std::string& appName ) ;

  // destructor
  ~test_decompress () ;

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
test_decompress::test_decompress ( const std::string& appName )
  : AppUtils::AppBase( appName )
  , m_inputArg("input-file", "path to the input file")
  , m_outputArg("output-file", "path to the output file")
  , m_input(0)
  , m_inputSize(0)
  , m_output(0)
  , m_outputSize(0)
{
  addArgument( m_inputArg ) ;
  addArgument( m_outputArg ) ;
}

//--------------
// Destructor --
//--------------
test_decompress::~test_decompress ()
{
  delete [] m_input;
  delete [] m_output;
}

/**
 *  Main method which runs the whole application
 */
int
test_decompress::runApp ()
{
  int stat = read();
  if (stat != 0) return stat;

  // open output file
  int fd = open(m_outputArg.value().c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666);
  if (fd < 0) {
    perror("failed to open output file");
    return errno;
  }

  // guess the size of output data
  m_outputSize = m_inputSize;
  m_output = new Bytef[m_outputSize];

  struct rusage ru0;
  stat = getrusage(RUSAGE_SELF, &ru0);


  z_stream stream;
  memset(&stream, 0, sizeof(stream));
  stream.next_in = m_input;
  stream.avail_in = m_inputSize;
  stream.next_out = m_output;
  stream.avail_out = m_outputSize;

  stat = inflateInit(&stream);
  if (stat != Z_OK) {
    std::cerr << "failed in inflateInit():" << stat << std::endl;
    return stat;
  }


  do {
    /* Uncompress some data */
    stat = inflate(&stream, Z_SYNC_FLUSH);

    std::cout << "inflate:  total_in: " << stream.total_in
        << " avail_in: " << stream.avail_in
        << " total_out: " << stream.total_out
        << " avail_out: " << stream.avail_out
        << std::endl;

    if (stat != Z_OK and stat != Z_STREAM_END) {
      std::cerr << "failed in inflate(): " << stat << std::endl;
      std::cerr << "  stream.total_in: " << stream.total_in << std::endl;
      std::cerr << "  stream.total_out: " << stream.total_out << std::endl;
      inflateEnd(&stream);
      return stat;
    }

    ::write(fd, m_output, m_outputSize-stream.avail_out);
    /* Update pointers to buffer for next set of uncompressed data */
    stream.next_out = m_output;
    stream.avail_out = m_outputSize;

  } while (stat == Z_OK);

  struct rusage ru1;
  stat = getrusage(RUSAGE_SELF, &ru1);
  std::cout << "user sec: " << (ru1.ru_utime.tv_sec - ru0.ru_utime.tv_sec) +
      (ru1.ru_utime.tv_usec - ru0.ru_utime.tv_usec)/1e6 << '\n';
  std::cout << "sys  sec: " << (ru1.ru_stime.tv_sec - ru0.ru_stime.tv_sec) +
      (ru1.ru_stime.tv_usec - ru0.ru_stime.tv_usec)/1e6 << '\n';

  close(fd);

  // return 0 on success, other values for error (like main())
  return 0 ;
}


int
test_decompress::read()
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

}


// this defines main()
APPUTILS_MAIN(pazlib::test_decompress)
