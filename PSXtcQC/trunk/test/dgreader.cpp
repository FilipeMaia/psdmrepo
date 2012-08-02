//===================

#include <stdlib.h>
#include <iostream> // for cout, puts etc.
#include <fcntl.h>  // open()
#include <unistd.h> // read()
//#include <sstream>  // for streamstring
#include <iomanip> // for setw

//#include <stdio.h>
//using namespace std;

#include "pdsdata/xtc/XtcFileIterator.hh"
#include "pdsdata/xtc/Dgram.hh"
//#include "pdsdata/xtc/Damage.hh"
//#include "pdsdata/xtc/TransitionId.hh"
//#include "MsgLogger/MsgLogger.h"

//#include "PSXtcQC/XtcQCIterator.h"
#include "XtcInput/XtcFileName.h"
#include "PSXtcQC/MyXtcQCIterator.h"
#include "PSXtcQC/QCStatistics.h"
#include "PSXtcQC/FileFinder.h"

//===================
// for iterate_over_dgrams_in_xstream
#include "XtcInput/ChunkFileIterList.h"
#include "XtcInput/XtcStreamDgIter.h"
#include <boost/make_shared.hpp>

using namespace XtcInput;
//===================

  //void atExit1 (void){ puts ("Exit function 1."); }
  //void atExit2 (void){ puts ("Exit function 2."); }
  //atexit (atExit1);
  //atexit (atExit2);
  //exit(1);
  //exit(2);

//===================

void usage(char* name) {
  fprintf(stderr,"Usage: %s [-f] <filename> [-h]\n", name);
}

//===================

std::string parse_input_parameters(int argc, char *argv[])
{
  // parse standard option-arguments:
  char* xtcname = 0;
  int   c;
  while ((c=getopt(argc, argv, ":f:h")) != -1) {
    switch (c) {
    case 'f' :
      xtcname = optarg;
      break;
    case 'h' :
      usage(argv[0]);
      //exit(0);
      break;
    case ':' :
      std::cout << "Missing argument\n";          
      usage(argv[0]);
      exit(0);
    case '?' :
      std::cout << "Non-defined option\n";          
      usage(argv[0]);
      exit(0);
    default:
      std::cout << "Default should never happen...";
      abort();
    }
  }

  // if a single additional argument is used without option "-f", assume that this is a file name...
  if (argc == 2) {
     xtcname = argv[1];
  }
  
  // if the file-name still is not defined 
  if (!xtcname) { 
      std::cout << "File name is not defined...\n";          
      usage(argv[0]); return std::string("file-name-is-non-defined");
  } // exit(2); }

  return std::string(xtcname);
}

//===================

void iterate_over_dgrams_in_xfile(std::string& xtcfname) // int fd)
{
  int fd = open(xtcfname.c_str(),O_RDONLY | O_LARGEFILE);    
  std::cout << "File descriptor: " << fd << "\n";

  Pds::XtcFileIterator iter(fd,0x1000000);
  Pds::Dgram* dg;
  unsigned long long dg_first_byte=0;
  unsigned ndgram=0;
  PSXtcQC::QCStatistics* qcstat= new PSXtcQC::QCStatistics();

  while ((dg = iter.next())) {                         // Iterate over all datagrams in file

    ndgram++;
    //if (ndgram > 100) break;
  
    qcstat->processDgram(dg,ndgram,dg_first_byte);              // Check everything for Dgram

    PSXtcQC::MyXtcQCIterator iterx(&(dg->xtc), qcstat, ndgram); // By default the top xtc in Dgram is assumed at 0-depth
    iterx.iterate();                                            // Begin iteration for top xtc container

    dg_first_byte += sizeof(*dg) + dg->xtc.sizeofPayload();
  }

  qcstat->printQCSummary(ndgram);

  close(fd);
}

//===================

void iterate_over_dgrams_in_xstream(FileFinder* ff)
{
  // Get vector with list of input chunk-files for one stream: 
  std::vector<std::string> v_names = ff->get_vect_of_chunks();
  std::vector<XtcFileName> file_names;
  // Copy vector for recasting...
  for(std::vector<std::string>::iterator it=v_names.begin(); it!=v_names.end(); it++)
    file_names.push_back( XtcFileName(*it) );

  boost::shared_ptr<ChunkFileIterI> fileItr =
    boost::make_shared<ChunkFileIterList>(file_names.begin(), file_names.end());

  XtcStreamDgIter dgIter(fileItr, 128*1024*1024);
  Pds::Dgram* dg;
  unsigned long long dg_first_byte=0;
  unsigned ndgram=0;
  PSXtcQC::QCStatistics* qcstat= new PSXtcQC::QCStatistics();

  while (true) {

    XtcInput::Dgram::ptr dgptr = dgIter.next();
    if (not dgptr) break;

    dg = dgptr.get();

    ndgram++; 
    //if (ndgram > 100) break;    

    qcstat->processDgram(dg,ndgram,dg_first_byte);              // Check everything for Dgram

    PSXtcQC::MyXtcQCIterator iterx(&(dg->xtc), qcstat, ndgram); // By default the top xtc in Dgram is assumed at 0-depth
    iterx.iterate();                                            // Begin iteration for top xtc container

    dg_first_byte += sizeof(*dg) + dg->xtc.sizeofPayload();
  }

  qcstat->printQCSummary(ndgram);
}

//===================

int main(int argc, char *argv[])
{
  std::string xtcfname = parse_input_parameters(argc, argv); 
  std::cout << "Input file name: " << xtcfname << "\n";

  //iterate_over_dgrams_in_xfile(xtcfname);

  FileFinder* ff = new FileFinder(xtcfname);
  iterate_over_dgrams_in_xstream(ff);

  return 0;
}

//===================
