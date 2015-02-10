//--------------------------
#include "PSQt/GeoImage.h"

#include <iostream>    // for std::cout
#include <fstream>     // for std::ifstream(fname)
//#include <math.h>  // atan2
//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

GeoImage::GeoImage(const std::string& fname_geo, const std::string& fname_img) 
  : m_fname_geo(fname_geo)
  , m_fname_img(fname_img)
{
  this -> check_fnames();

  std::cout << "GeoImage::GeoImage(string) :" 
            << "\n  fname_geo: " << m_fname_geo  
            << "\n  fname_img: " << m_fname_img  
            << '\n';

  m_geometry = new PSCalib::GeometryAccess(m_fname_geo);
  m_pix_scale_size = m_geometry->get_pixel_scale_size();
  m_size = m_geometry->get_top_geo()->get_size_geo_array();

  std::cout << "  pix_scale_size: " << m_pix_scale_size
            << "  size          : " << m_size
            << '\n';

  unsigned shape[] = {m_size};  

  m_ndaio = new GeoImage::NDAIO(fname_img, shape, 1, 0);
  //m_ndaio->print_ndarray();
  m_anda = m_ndaio->get_ndarray(); // or get_ndarray(fname);  
}

//--------------------------

void
GeoImage::check_fnames()
{
  const std::string base_dir = "/reg/g/psdm/detector/alignment/cspad/calib-cxi-ds1-2014-05-15/";
  const std::string fname_geo = base_dir + "calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data"; 
  const std::string fname_nda = base_dir + "cspad-arr-cxid2714-r0023-lysozyme-rings.txt"; 

  if (m_fname_geo == std::string()) m_fname_geo = fname_geo;
  if (m_fname_img == std::string()) m_fname_img = fname_nda;
}

//--------------------------

const ndarray<const PSCalib::GeometryAccess::image_t, 2>
GeoImage::get_image()
{
  const unsigned* iX;
  const unsigned* iY;
  unsigned        isize;

  m_geometry->get_pixel_coord_indexes(iX, iY, isize);
  //m_geometry.get_pixel_coord_indexes(iX, iY, isize, ioname, ioindex, pix_scale_size_um, xy0_off_pix);

  return PSCalib::GeometryAccess::img_from_pixel_arrays(iX, iY, &m_anda[0], isize);
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------

} // namespace PSQt

//--------------------------


