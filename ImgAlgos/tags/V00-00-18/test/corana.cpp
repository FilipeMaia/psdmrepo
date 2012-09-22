//===================

#include "ImgAlgos/CorAnaInputParameters.h"
#include "ImgAlgos/CorAnaData.h"

//===================
using namespace ImgAlgos;
//===================

int main(int argc, char *argv[])
{
  INPARS->parse_input_parameters(argc, argv); // Instantiate singleton
  //CorAnaData* cad = new CorAnaData();       // Work on data  
  CorAnaData cad;                             // Work on data  
  INPARS->close_log_file();                   // Close log file
  return 0;
}

//===================
