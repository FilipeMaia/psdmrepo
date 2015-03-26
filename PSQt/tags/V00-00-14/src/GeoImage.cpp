//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------

//--------------------------
#include "PSQt/GeoImage.h"
#include "PSQt/Logger.h"
#include "PSQt/QGUtils.h"

#include <iostream>    // for std::cout
#include <fstream>     // for std::ifstream(fname)
#include <cstdlib>     // for rand()
#include <cstring> // for memcpy

//#include <math.h>  // atan2
//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

GeoImage::GeoImage(PSCalib::GeometryAccess* geometry, const std::string& fname_img)
  : QObject(NULL)
  , m_fname_geo(std::string())
  , m_fname_img(fname_img)
  , m_geometry(geometry)
  , m_ndaio(0)
{
  this -> setGeometryPars();
  this -> loadImageArrayFromFile(m_fname_img);
  this -> connectTestSignalsSlots();
}

//--------------------------

GeoImage::GeoImage(const std::string& fname_geo, const std::string& fname_img) 
  : QObject(NULL)
  , m_fname_geo(fname_geo)
  , m_fname_img(fname_img)
  , m_geometry(0)
  , m_ndaio(0)
{
  this -> checkFileNames();
  this -> loadGeometryFromFile(m_fname_geo);
  this -> setGeometryPars();
  this -> loadImageArrayFromFile(m_fname_img);
  this -> connectTestSignalsSlots();
}

//--------------------------
void
GeoImage::connectTestSignalsSlots()
{
  connect(this, SIGNAL(imageIsUpdated(ndarray<GeoImage::raw_image_t,2>&)), 
          this, SLOT(testSignalImageIsUpdated(ndarray<GeoImage::raw_image_t,2>&)));
}

//--------------------------
void
GeoImage::checkFileNames()
{
  const std::string base_dir = "/reg/g/psdm/detector/alignment/cspad/calib-cxi-ds1-2014-05-15/";
  const std::string fname_geo = base_dir + "calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data"; 
  const std::string fname_nda = base_dir + "cspad-arr-cxid2714-r0023-lysozyme-rings.txt"; 

  if (m_fname_geo.empty()) m_fname_geo = fname_geo;
  if (m_fname_img.empty()) m_fname_img = fname_nda;

  MsgInLog(_name_(), INFO, "Use fname_geo: " + m_fname_geo); 
  MsgInLog(_name_(), INFO, "Use fname_img: " + m_fname_img); 
}

//--------------------------
void
GeoImage::loadGeometryFromFile(const std::string& fname_geo)
{
  m_fname_geo = fname_geo;

  if (m_geometry) delete m_geometry;
  m_geometry = new PSCalib::GeometryAccess(m_fname_geo);
  MsgInLog(_name_(), INFO, "Geometry is loaded from file" + m_fname_geo); 
}

//--------------------------
void
GeoImage::setGeometryPars()
{
  m_pix_scale_size = m_geometry->get_pixel_scale_size();
  m_size = m_geometry->get_top_geo()->get_size_geo_array();

  stringstream ss; 
  ss << "Set geometry parameters: pix_scale_size[um] = " << m_pix_scale_size
     << "  size = " << m_size;
  MsgInLog(_name_(), DEBUG, ss.str()); 
}

//--------------------------
void
GeoImage::loadImageArrayFromFile(const std::string& fname)
{
  m_fname_img = fname;

  if (m_ndaio) delete m_ndaio;
  unsigned shape[] = {m_size};  
  m_ndaio = new GeoImage::NDAIO(m_fname_img, shape, 1, 0);
  //m_ndaio->print_ndarray();
  m_anda = m_ndaio->get_ndarray(); // or get_ndarray(fname);  
  //std::cout << m_anda << '\n';
  stringstream ss; 
  ss << "Image array of size " << m_anda.size() << " is loaded from file" << m_fname_img;
  MsgInLog(_name_(), INFO,  ss.str()); 
}

//--------------------------
//--------------------------
//--------- SLOTS ----------
//--------------------------
//--------------------------
void 
GeoImage::onGeometryIsLoaded(PSCalib::GeometryAccess* geometry)
{
  MsgInLog(_name_(), INFO, "onGeometryIsLoaded() - begin update"); 

  if (m_geometry) delete m_geometry;
  m_geometry = geometry;


  this -> setGeometryPars();

  this -> updateImage();
}

//--------------------------
void 
GeoImage::onGeoIsChanged(shpGO& geo)
{
  //std::cout << "testSignalGeoIsChanged():\n";
  //geo->print_geo();
  //m_geo->print_geo();
  MsgInLog(_name_(), DEBUG, "onGeoIsChanged(): " + geo->str_data()); // string_geo()); 

  //m_geometry->print_list_of_geos();

  this -> updateImage();
}

//--------------------------
void 
GeoImage::onImageFileNameIsChanged(const std::string& fname)
{
  MsgInLog(_name_(), INFO, "onImageFileNameIsChanged(): fname = " + fname); 

  this -> loadImageArrayFromFile(fname);
  updateImage();
}

//--------------------------
void 
GeoImage::updateImage()
{
  MsgInLog(_name_(), DEBUG, "updateImage(): re-generate raw image ndarray and emit signal: imageIsUpdated(nda)"); 

  //const ndarray<GeoImage::image_t,2> nda = getRandomImage();
  //const ndarray<GeoImage::image_t, 2> nda = getNormalizedImage();
  //emit normImageIsUpdated(nda);

  //ndarray<const GeoImage::raw_image_t, 2> nda = getImage();
  ndarray<GeoImage::raw_image_t, 2>& nda = getImage();
  emit imageIsUpdated(nda);
}

//--------------------------
void 
GeoImage::setFirstImage()
{
  this -> updateImage();
}

//--------------------------

//const ndarray<const GeoImage::raw_image_t,2>
ndarray<GeoImage::raw_image_t,2> &
GeoImage::getImage()
{
  const unsigned* iX;
  const unsigned* iY;
  unsigned        isize;

  m_geometry->get_top_geo()->evaluate_pixel_coords(true,true);
  double pix_scale_size =m_geometry->get_pixel_scale_size ();
  const int xy0_off_pix[2] = {1000,1000};
  //m_geometry->get_pixel_coord_indexes(iX, iY, isize);
  //m_geometry->get_pixel_coord_indexes(iX, iY, isize, ioname, ioindex, pix_scale_size_um, xy0_off_pix);
  m_geometry->get_pixel_coord_indexes(iX, iY, isize, std::string(), 0, pix_scale_size, xy0_off_pix);

  //return PSCalib::GeometryAccess::img_from_pixel_arrays(iX, iY, &m_anda[0], isize);
  return m_geometry->ref_img_from_pixel_arrays(iX, iY, &m_anda[0], isize);
}

//--------------------------
//--------------------------
//------ For tests ---------
//--------------------------
//--------------------------
void 
GeoImage::testSignalImageIsUpdated(ndarray<GeoImage::raw_image_t,2>& nda)
{  
  stringstream ss; ss << "testSignalImageIsUpdated(), size = " << nda.size();
  MsgInLog(_name_(), DEBUG, ss.str()); 
}

//--------------------------

ndarray<GeoImage::image_t,2>
GeoImage::getNormalizedImage()
{
  typedef PSCalib::GeometryAccess::image_t raw_image_t; // double
  typedef GeoImage::image_t image_t;                   // uint32_t

  static unsigned counter = 0;
  stringstream ss; ss << "getNormalizedImage() " << ++counter;
  MsgInLog(_name_(), INFO, ss.str());

  const ndarray<const raw_image_t, 2> dnda = this->getImage();

  return getUint32NormalizedImage<const raw_image_t>(dnda); // from QGUtils
}

//--------------------------

ndarray<GeoImage::image_t,2>
GeoImage::getRandomImage()
{
  const unsigned rows = 1750;
  const unsigned cols = 1750;
  float hue1 = rand()%360; // 0;
  float hue2 = hue1 + 360;
  stringstream ss; ss << "Color bar image for hue angle range " << hue1 << " : " << hue2;
  MsgInLog(_name_(), INFO, ss.str());
  return getColorBarImage(rows, cols, hue1, hue2); // from QGUtils
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------

} // namespace PSQt

//--------------------------


