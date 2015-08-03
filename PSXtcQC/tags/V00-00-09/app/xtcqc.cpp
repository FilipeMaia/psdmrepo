//===================

#include <stdlib.h>
#include <fstream> // 
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

#include "PSXtcQC/InputParameters.h"

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

void iterate_over_dgrams_in_xfile(std::string& xtcfname) // int fd)
{
  int fd = open(xtcfname.c_str(),O_RDONLY | O_LARGEFILE);    
  std::cout << "File descriptor: " << fd << "\n";

  Pds::XtcFileIterator iter(fd,0x1000000);
  Pds::Dgram* dg;
  unsigned long long dg_first_byte=0;
  unsigned ndgram=0;
  PSXtcQC::QCStatistics* qcstat= new PSXtcQC::QCStatistics(INPARS->get_ostream());

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

//void iterate_over_dgrams_in_xstream(FileFinder* ff)

void iterate_over_dgrams_in_xstream(std::vector<std::string>& v_names)
{
  // Get vector with list of input chunk-files for one stream: 
  // std::vector<std::string> v_names = ff->get_vect_of_chunks();
  std::vector<XtcFileName> file_names;
  // Copy vector for recasting...
  for(std::vector<std::string>::iterator it=v_names.begin(); it!=v_names.end(); it++)
    file_names.push_back( XtcFileName(*it) );

  boost::shared_ptr<ChunkFileIterI> fileItr =
    boost::make_shared<ChunkFileIterList>(file_names.begin(), file_names.end());

  XtcStreamDgIter dgIter(fileItr);
  Pds::Dgram* dg;
  unsigned long long dg_first_byte=0;
  unsigned ndgram=0;
  PSXtcQC::QCStatistics* qcstat= new PSXtcQC::QCStatistics(INPARS->get_ostream());

  while (true) {

    XtcInput::Dgram dgram = dgIter.next();
    if (not dgram.dg()) break;

    dg = dgram.dg().get();

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
  INPARS->parse_input_parameters(argc, argv); 

  //std::string xtcfname = "non-defined-file";
  //iterate_over_dgrams_in_xfile(xtcfname);
  //FileFinder* ff = new FileFinder(xtcfname);
  //iterate_over_dgrams_in_xstream(ff);

  iterate_over_dgrams_in_xstream(INPARS->get_vector_fnames());

  INPARS->close_log_file();
  return 0;
}

//===================
