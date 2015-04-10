//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test class GeometryObject, GeometryAccess of the PSCalib packadge
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "PSCalib/GeometryObject.h"
#include "PSCalib/GeometryAccess.h"

#include "ImgAlgos/GlobalMethods.h" // for test ONLY
#include "pdscalibdata/NDArrIOV1.h"
 
#include <string>
#include <iostream>
#include <iomanip>  // for setw, setfill

//using std::cout;
//using std::endl;

using namespace std;
//using namespace PSCalib;

//-----------------
//-----------------
 
void test_cspad2x2 (int argc, char* argv[])
{
  //MecTargetChamber.0:Cspad2x2.1 
    string basedir = "/reg/neh/home1/dubrovin/LCLS/CSPad2x2Alignment/calib-cspad2x2-01-2013-02-13/";  
    string fname_geometry = basedir + "calib/CsPad2x2::CalibV1/MecTargetChamber.0:Cspad2x2.1/geometry/0-end.data";
    string fname_data     = basedir + "cspad2x2.1-ndarr-ave-meca6113-r0028.dat";

    typedef pdscalibdata::NDArrIOV1<double,3> NDAIO;
    unsigned shape[] = {185,388,2};
    NDAIO* arrio = new NDAIO(fname_data, shape, 0, 0); // 0377
    //arrio->print_ndarray();
    
    PSCalib::GeometryAccess geometry(fname_geometry, 0); // 0177777 - prints more details for debugging
    geometry.print_list_of_geos_children();

    cout << "\n\ntest_cspad2x2(...):\n";

    const unsigned * iX;
    const unsigned * iY;
    unsigned   isize;
    geometry.get_pixel_coord_indexes(iX, iY, isize);

    ndarray<PSCalib::GeometryAccess::image_t, 2> img = 
      PSCalib::GeometryAccess::img_from_pixel_arrays(iX, iY, arrio->get_ndarray().data(), isize);
    cout << img << "\n";  

    string fname_img("cspad2x2-img-test.txt");
    cout << "    argc=" << argc << ", if (argc>1) - save image in text file " << fname_img << '\n';
    if (argc>1) ImgAlgos::save2DArrayInFile<PSCalib::GeometryAccess::image_t>(fname_img.c_str(), img, true);
}

//-----------------
//-----------------
 
int main (int argc, char* argv[])
{
  std::string pname("SOME_PARENT");
  unsigned    pindex=1;
  std::string oname("SENS2X1:V1");
  unsigned    oindex=2;
  double      x0=3;
  double      y0=4;
  double      z0=5;
  double      rot_z=6;
  double      rot_y=7;
  double      rot_x=8;
  double      tilt_z=9;
  double      tilt_y=10;
  double      tilt_x=11;

  //-----------------
  cout << "Run " << argv[0] << '\n';     
  cout << "\n\nTest of PSCalib::GeometryObject\n";     
  PSCalib::GeometryObject* geo = new PSCalib::GeometryObject(pname, 
							       pindex,
							       oname, 
							       oindex,
							       x0,    
							       y0,    
							       z0,    
							       rot_z, 
							       rot_y, 
							       rot_x, 
							       tilt_z,
							       tilt_y,
							       tilt_x );

  geo->print_geo();

  cout << "Size of geo: " << geo->get_size_geo_array() << "\n"; 

  //-----------------
  cout << "\n\nTest of PSCalib::GeometryAccess\n";     

  //string basedir = "/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-03-19/";
  //string fname_geometry = basedir + "calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/0-end.data";
  //string fname_data     = basedir + "cspad-ndarr-ave-cxii0114-r0227.dat";

  string basedir = "/reg/neh/home1/dubrovin/LCLS/CSPadAlignment-v01/calib-cxi-ds1-2014-05-15/";
  string fname_geometry = basedir + "calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data";
  string fname_data     = basedir + "cspad-arr-cxid2714-r0023-lysozyme-rings.txt";
 
  // Temporary solution for Summer 2014 shutdown:
  // string basedir = "/home/pcds/LCLS/calib/geometry/";
  // string fname_geometry = basedir + "2-end.data";
  // string fname_data     = basedir + "cspad-arr-cxid2714-r0023-lysozyme-rings.txt";
 

  unsigned print_bits = 0; // 0377;
  PSCalib::GeometryAccess geometry(fname_geometry, print_bits);

  //-----------------
  cout << "\n\nTest of print_list_of_geos():\n";  
  geometry.print_list_of_geos();

  //-----------------
  cout << "\n\nTest of print_list_of_geos_children():\n";  
  geometry.print_list_of_geos_children();

  //-----------------
  cout << "\n\nTest of print_comments_from_dict():\n";  
  geometry.print_comments_from_dict();

  //-----------------
  cout << "\n\nTest of get_geo(...):\n";  
  geometry.get_geo("QUAD:V1", 1)->print_geo_children();

  //-----------------
  cout << "\n\nTest of get_top_geo():\n";  
  geometry.get_top_geo()->print_geo_children();

 //-----------------
  cout << "\n\nTest of print_pixel_coords(...) for quad:\n";
  geometry.print_pixel_coords("QUAD:V1", 1);
  cout << "\n\nTest of print_pixel_coords(...) for CSPAD:\n";
  geometry.print_pixel_coords();

  //-----------------
  cout << "\n\nTest of get_pixel_coords(...):\n";
  const double* X;
  const double* Y;
  const double* Z;
  unsigned   size;
  geometry.get_pixel_coords(X,Y,Z,size);

  cout << "size=" << size << '\n' << std::fixed << std::setprecision(1);  
  cout << "X: "; for(unsigned i=0; i<10; ++i) cout << std::setw(10) << X[i] << ", "; cout << "...\n"; 
  cout << "Y: "; for(unsigned i=0; i<10; ++i) cout << std::setw(10) << Y[i] << ", "; cout << "...\n"; 
  cout << "Z: "; for(unsigned i=0; i<10; ++i) cout << std::setw(10) << Z[i] << ", "; cout << "...\n"; 

  //-----------------
  cout << "\n\nTest of get_pixel_coord_indexes(...):\n";
  const unsigned * iX;
  const unsigned * iY;
  unsigned   isize;

  const std::string ioname = "QUAD:V1";
  const unsigned ioindex = 1;
  const double pix_scale_size_um = 109.92;
  const int xy0_off_pix[] = {200,200};
  geometry.get_pixel_coord_indexes(iX, iY, isize, ioname, ioindex, pix_scale_size_um, xy0_off_pix);

  cout << "QUAD size=" << isize << '\n' << std::fixed << std::setprecision(1);  
  cout << "iX: "; for(unsigned i=0; i<10; ++i) cout << std::setw(10) << iX[i] << ", "; cout << "...\n"; 
  cout << "iY: "; for(unsigned i=0; i<10; ++i) cout << std::setw(10) << iY[i] << ", "; cout << "...\n"; 

  geometry.get_pixel_coord_indexes(iX, iY, isize);

  cout << "CSPAD size=" << isize << '\n' << std::fixed << std::setprecision(1);  
  cout << "iX: "; for(unsigned i=0; i<10; ++i) cout << std::setw(10) << iX[i] << ", "; cout << "...\n"; 
  cout << "iY: "; for(unsigned i=0; i<10; ++i) cout << std::setw(10) << iY[i] << ", "; cout << "...\n"; 

  //-----------------

  cout << "\n\nTest of PSCalib::img_from_pixel_arrays(...):\n";
  ndarray<PSCalib::GeometryAccess::image_t, 2> img = 
          PSCalib::GeometryAccess::img_from_pixel_arrays(iX, iY, 0, isize);
  cout << img << "\n";  
  cout << "    argc=" << argc << ", if (argc>1) - save image in text file.\n";
  if (argc>1) ImgAlgos::save2DArrayInFile<PSCalib::GeometryAccess::image_t>("cspad-img-test.txt", img, true);
                   
 //-----------------
  cout << "\n\nTest of get_pixel_areas(...):\n";
  const double* A;
  geometry.get_pixel_areas(A,size);

  cout << "size=" << size << '\n' << std::fixed << std::setprecision(1);  
  cout << "Areas: "; for(unsigned i=190; i<200; ++i) cout << std::setw(10) << A[i] << ", "; cout << "...\n"; 

 //-----------------
  cout << "\n\nTest of get_pixel_mask(...):\n";
  const int* mask;
  unsigned mbits = 077;
  geometry.get_pixel_mask(mask, size, std::string(), 0, mbits);

  cout << "size=" << size << '\n' << std::fixed << std::setprecision(1);  
  cout << "Mask[row=0]: "; for(unsigned i=190; i<200; ++i) cout << std::setw(10) << mask[i]     << ", "; cout << "...\n"; 
  cout << "Mask[row=1]: "; for(unsigned i=190; i<200; ++i) cout << std::setw(10) << mask[i+388] << ", "; cout << "...\n"; 

  //-----------------
  cout << "\n\nTest of get_size_geo_array(...) for";
  cout << "\nQuad : " << geometry.get_geo("QUAD:V1", 1)->get_size_geo_array();
  cout << "\nCSPAD: " << geometry.get_top_geo()->get_size_geo_array();
 
  //-----------------
  cout << "\n\nTest of get_pixel_scale_size() for" << std::setprecision(2);
  cout << "\nQuad     : " << geometry.get_geo("QUAD:V1", 1)->get_pixel_scale_size();
  cout << "\nCSPAD    : " << geometry.get_top_geo()->get_pixel_scale_size();
  cout << "\ngeometry : " << geometry.get_pixel_scale_size();
 
  //-----------------
  cout << "\n\nTest of get_dict_of_comments():\n";
  std::map<std::string, std::string>& dict = geometry.get_dict_of_comments();
  cout << "dict['HDR'] = " << dict["HDR"] << '\n';

  //-----------------
  cout << "\n\nTest of save_pars_in_file(path):\n";
  geometry.save_pars_in_file("test.txt");

  //-----------------
  cout << "\n\nTest of save_pars_in_file(path):\n";

  const std::string fname_test("test.txt");
  geometry.set_print_bits(1+64);
  geometry.save_pars_in_file(fname_test);

  //-----------------
  cout << "\n\nTest of load_pars_from_file(path):\n";
  geometry.set_print_bits(1);
  geometry.load_pars_from_file(fname_test);
  geometry.print_list_of_geos();


  //-----------------
  cout << "\n\nTest cspad2x2:\n";
  test_cspad2x2(argc, argv);

  //-----------------
  cout << "End of " << argv[0] << '\n';
  return 0;
}

//-----------------
