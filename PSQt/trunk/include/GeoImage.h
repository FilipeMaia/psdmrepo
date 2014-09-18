#ifndef GEOIMAGE_H
#define GEOIMAGE_H
//--------------------------

#include "PSCalib/GeometryAccess.h"
#include "ndarray/ndarray.h"

#include "pdscalibdata/NDArrIOV1.h"
 
namespace PSQt {
//--------------------------

class GeoImage
{
 public:
    typedef pdscalibdata::NDArrIOV1<double,1> NDAIO;

    GeoImage(const std::string& fname_geo, 
             const std::string& fname_img); 

    const ndarray<const PSCalib::GeometryAccess::image_t, 2> get_image();

 protected:
    //void setFrame() ;

 private:
    PSCalib::GeometryAccess* m_geometry;

    std::string m_fname_geo;
    std::string m_fname_img;
    double      m_pix_scale_size;
    unsigned    m_size;       
    NDAIO*      m_ndaio;

    ndarray<const double, 1> m_anda;
};

//--------------------------
} // namespace PSQt

#endif // GEOIMAGE_H
//--------------------------
