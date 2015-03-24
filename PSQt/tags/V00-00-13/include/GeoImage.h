#ifndef GEOIMAGE_H
#define GEOIMAGE_H
//--------------------------

#include "pdscalibdata/NDArrIOV1.h"
#include "PSCalib/GeometryAccess.h"
#include "ndarray/ndarray.h"
#include <stdint.h> // uint8_t, uint32_t, etc.

#include <QObject>
 
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

class GeoImage : public QObject
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:

    typedef PSCalib::GeometryAccess::image_t raw_image_t; // double
    typedef uint32_t image_t;

    typedef boost::shared_ptr<PSCalib::GeometryObject> shpGO;
    typedef pdscalibdata::NDArrIOV1<double,1> NDAIO;

    GeoImage(PSCalib::GeometryAccess*,
             const std::string& fname_img = std::string()); 

    GeoImage(const std::string& fname_geo = std::string(), 
             const std::string& fname_img = std::string()); 

    void setFirstImage();

    //const ndarray<const raw_image_t, 2> getImage();
    ndarray<raw_image_t, 2>& getImage();

    ndarray<image_t,2> getNormalizedImage();
    ndarray<image_t,2> getRandomImage();

 public slots:
    void onGeometryIsLoaded(PSCalib::GeometryAccess*);
    void onGeoIsChanged(shpGO& geo);
    void onImageFileNameIsChanged(const std::string& str);
    void testSignalImageIsUpdated(ndarray<GeoImage::raw_image_t,2>&);

 signals :
    void imageIsUpdated(ndarray<GeoImage::raw_image_t,2>&);

 private:
    std::string m_fname_geo;
    std::string m_fname_img;
    PSCalib::GeometryAccess* m_geometry;
    ndarray<const double, 1> m_anda;

    double      m_pix_scale_size;
    unsigned    m_size;       
    NDAIO*      m_ndaio;

    //uint32_t*   m_img;
    //ndarray<uint32_t, 2> m_nda_img;
    //ndarray<const raw_image_t, 2>& p_nda_img;

    inline const char* _name_(){return "GeoImage";}
    void connectTestSignalsSlots();
    void checkFileNames();
    void loadGeometryFromFile(const std::string& fname_geo);
    void setGeometryPars();
    void loadImageArrayFromFile(const std::string& fname_img);
    void updateImage();
};

//--------------------------
} // namespace PSQt

#endif // GEOIMAGE_H
//--------------------------
