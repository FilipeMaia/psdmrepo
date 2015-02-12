#ifndef GEOIMAGE_H
#define GEOIMAGE_H
//--------------------------

#include "pdscalibdata/NDArrIOV1.h"
#include "PSCalib/GeometryAccess.h"
#include "ndarray/ndarray.h"
 
namespace PSQt {

//--------------------------

/**
 *  @ingroup PSQt
 *
 *  @brief GeoImage - generates image using geometry and ndarray of intensity
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see GUIMain
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

//--------------------------

class GeoImage
{
 public:
    typedef pdscalibdata::NDArrIOV1<double,1> NDAIO;

    GeoImage(const std::string& fname_geo = std::string(), 
             const std::string& fname_img = std::string()); 

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

    void check_fnames();
    inline const char* _name_(){return "GeoImage";}
};

//--------------------------
} // namespace PSQt

#endif // GEOIMAGE_H
//--------------------------
