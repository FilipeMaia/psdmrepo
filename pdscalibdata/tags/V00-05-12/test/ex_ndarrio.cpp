//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//      $Revision$
//
// Description:
//      Test class NDArrIOV1 of the pdscalibdata packadge
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "pdscalibdata/NDArrIOV1.h"

#include <string>
#include <iostream>

//using std::cout;
//using std::endl;
using namespace std;

int main ()
{
  typedef pdscalibdata::NDArrIOV1<float,3> NDAIO;

  //---------------------------
  cout << "Test of pdscalibdata::NDArrIOV1\n";     

  const std::string fname("/reg/neh/home1/dubrovin/LCLS/PSANA-V01/pdscalibdata/test/test.data");
  NDAIO *ndaio1 = new NDAIO(fname);  

  cout << "PRINT TESTS:\n";

  ndaio1->print();
  ndaio1->print_file();
  ndaio1->print_ndarray();

  //const ndarray<const float,3>& nda = ndaio1->get_ndarray(fname);
  const ndarray<const float,3>& nda = ndaio1->get_ndarray();
  cout << "Test local print:\n" << nda << '\n';


  //---------------------------
  cout << "Test save_ndarray:\n" << nda << '\n';

  std::vector<std::string> v_comments;
  v_comments.push_back("TITLE      File to load ndarray of calibration parameters");
  v_comments.push_back("");
  v_comments.push_back("EXPERIMENT amo12345");
  v_comments.push_back("DETECTOR   Camp.0:pnCCD.1");
  v_comments.push_back("CALIB_TYPE pedestals");

  std::string fname_test("zzz.txt");
  NDAIO::save_ndarray(nda, "zzz.txt", v_comments);  

  //---------------------------
  cout << "Test ndarray from file: " << fname_test << '\n';

  NDAIO *ndaio2 = new NDAIO(fname_test);  
  ndaio2->print_ndarray();

  return 0;
}
