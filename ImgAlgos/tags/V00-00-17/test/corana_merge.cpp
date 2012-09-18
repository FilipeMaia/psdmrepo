//===================

#include "ImgAlgos/CorAnaInputParameters.h"
#include "ImgAlgos/CorAnaMergeFiles.h"

//===================
using namespace ImgAlgos;
//===================

int main(int argc, char *argv[])
{
  INPARS->parse_input_parameters(argc, argv); // Instantiate singleton
  CorAnaMergeFiles merge;                     // Work on data  
  INPARS->close_log_file();                   // Close log file
  return 0;
}

//===================
