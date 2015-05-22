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

typedef pdscalibdata::NDArrIOV1<float,3> NDAIO;

int main ()
{
  //---------------------------
  cout << "Test of pdscalibdata::NDArrIOV1\n";     

  const std::string fname("/reg/neh/home1/dubrovin/LCLS/PSANA-V01/pdscalibdata/test/test.data");
  unsigned int shape[] = {3,4,8};
  NDAIO *ndaio1 = new NDAIO(fname,shape);  

  cout << "\nPRINT TESTS:\n";

  ndaio1->print();
  ndaio1->print_file();
  ndaio1->print_ndarray();

  //const ndarray<const float,3>& nda = ndaio1->get_ndarray(fname);
  const ndarray<const float,3>& nda = ndaio1->get_ndarray();
  cout << "\nTest local print:\n" << nda << '\n';

  //---------------------------
  std::string fname_test("/tmp/test_ndarrio_m.txt");
  cout << "\nTest save_ndarray:\n" << nda 
       << "\nin file: " << fname_test << '\n';
  std::vector<std::string> v_comments;
  v_comments.push_back("TITLE      File to load ndarray of calibration parameters");
  v_comments.push_back("");
  v_comments.push_back("EXPERIMENT amo12345");
  v_comments.push_back("DETECTOR   Camp.0:pnCCD.1");
  v_comments.push_back("CALIB_TYPE pedestals");
  NDAIO::save_ndarray(nda, fname_test, v_comments);  

  //---------------------------
  cout << "\nTest read ndarray from file: " << fname_test << '\n';
  NDAIO *ndaio2 = new NDAIO(fname_test, shape);  
  ndaio2->print_ndarray();

  //---------------------------
  const std::string fname_bwc("/reg/neh/home1/dubrovin/LCLS/PSANA-V01/pdscalibdata/test/test_bwc.data");
  cout << "\nBackward compatability test read ndarray from file: " << fname_bwc << '\n';
  unsigned int shape_bwc[] = {2,3,4};
  NDAIO *ndaio3 = new NDAIO(fname_bwc, shape_bwc);  
  ndaio3->print_ndarray();

  //---------------------------
  cout << "\nTest read ndarray from file with unknown shape: " << fname_test << '\n';
  NDAIO *ndaio4 = new NDAIO(fname_test);  
  ndaio4->print_ndarray();

  return 0;
}
