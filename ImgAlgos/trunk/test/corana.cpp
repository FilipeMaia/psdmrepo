//===================

#include <stdlib.h>
#include <fstream> // 
#include <iostream> // for cout, puts etc.
#include <fcntl.h>  // open()
#include <unistd.h> // read()
#include <iomanip> // for setw
//#include <sstream>  // for stringstream

//#include <stdio.h>
//using namespace std;
//#include "MsgLogger/MsgLogger.h"
//#include "ImgAlgos/FileFinder.h"
#include "ImgAlgos/CorAnaInputParameters.h"
#include "ImgAlgos/CorAnaData.h"

//===================
//#include <boost/make_shared.hpp>

using namespace ImgAlgos;
//===================

  //void atExit1 (void){ puts ("Exit function 1."); }
  //void atExit2 (void){ puts ("Exit function 2."); }
  //atexit (atExit1);
  //atexit (atExit2);
  //exit(1);
  //exit(2);

//===================

/*
void proc_file(std::string& fname) // int fd)
{
  int fd = open(fname.c_str(),O_RDONLY | O_LARGEFILE);    
  std::cout << "File descriptor: " << fd << "\n";
  close(fd);
}
*/

//===================
//===================

int main(int argc, char *argv[])
{
  INPARS->parse_input_parameters(argc, argv); // Instantiate singleton
  CorAnaData* cad = new CorAnaData();         // Work on data  
  INPARS->close_log_file();                   // Close log file
  return 0;
}

//===================
//===================
//===================
