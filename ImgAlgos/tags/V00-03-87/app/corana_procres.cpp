//===================
// The main module which serves as an example of how to get g2 data
//    g2(tau) = <I(t)*I(t+tau)> / (<I(t)> * <I(t+tau)>)
//

#include "ImgAlgos/CorAnaInputParameters.h"
#include "ImgAlgos/CorAnaProcResults.h"

//===================
using namespace ImgAlgos;
//===================

int main(int argc, char *argv[])
{
  INPARS->parse_input_parameters(argc, argv); // Instantiate singleton
  CorAnaProcResults proc_results;             // Work on results  
  INPARS->close_log_file();                   // Close log file
  return 0;
}

//===================
