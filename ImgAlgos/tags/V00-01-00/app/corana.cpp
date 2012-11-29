//===================
// The main module for calculation of the image (part) auto-correlation function
//    g2(tau) = <I(t)*I(t+tau)> / (<I(t)> * <I(t+tau)>)
//

#include "ImgAlgos/CorAnaInputParameters.h"
#include "ImgAlgos/CorAnaData.h"

//===================
using namespace ImgAlgos;
//===================

int main(int argc, char *argv[])
{
  INPARS->parse_input_parameters(argc, argv); // Instantiate singleton
  CorAnaData cad;                             // Work on data  
  INPARS->close_log_file();                   // Close log file
  return 0;
}

//===================
