#ifndef IMAGEPROC_H
#define IMAGEPROC_H
//--------------------------

#include "PSQt/GeoImage.h"
#include "ndarray/ndarray.h"
#include <stdint.h> // uint8_t, uint32_t, etc.

#include <QObject>
#include <QPointF>
#include <QMouseEvent>
 
namespace PSQt {

//--------------------------

/**
 *  @ingroup PSQt
 *
 *  @brief Processing of raw image ndarray after zoom or intensity range selection
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see GUIImageViewer
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

//--------------------------

class ImageProc : public QObject
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:

    typedef PSQt::GeoImage::raw_image_t raw_image_t; // double

    ImageProc(); 
    virtual ~ImageProc();

 public slots:
    void onImageIsUpdated(ndarray<GeoImage::raw_image_t,2>&);
    void onZoomIsChanged(int&, int&, int&, int&, float&, float&);
    void onCenterIsChanged(const QPointF&);
    //void onCenterIsMoved(const QPointF&); // signal on each motion of coursor
    //void onPressOnAxes(QMouseEvent*, QPointF);
    void testSignalRHistIsFilled(ndarray<float, 1>&, const unsigned&, const unsigned&);
    void testSignalSHistIsFilled(float*, const float&, const float&, const unsigned&);

    //const ndarray<const raw_image_t, 2> getImage();
 signals:
    void rhistIsFilled(ndarray<float, 1>&, const unsigned&, const unsigned&);
    void shistIsFilled(float*, const float&, const float&, const unsigned&);


 private:
    inline const char* _name_(){return "ImageProc";}
    void evaluateRadialIndexes();
    void fillRadialHistogram();
    void fillSpectralHistogram(const float& amin=0, const float& amax=0, const unsigned& nbins=100);
    void getIntensityLimits(float& amin, float& amax);

    float m_rbin_width;
    bool m_image_is_set;
    bool m_center_is_set;
    bool m_zoom_is_set;
    bool m_rindx_is_set;
    bool m_rhist_is_set;
    bool m_shist_is_set;

    unsigned m_ixmax;
    unsigned m_iymax;
    unsigned m_irmax;
    unsigned m_zirmin;
    unsigned m_zirmax;

    int m_zxmin;
    int m_zymin;
    int m_zxmax;
    int m_zymax;

    float    m_amin;
    float    m_amax;
    unsigned m_nbins;

    QPointF m_center;
    ndarray<GeoImage::raw_image_t,2> m_nda_image; 
    ndarray<unsigned,2> m_nda_rindx;
    ndarray<float, 1> m_nda_rhist;
    //ndarray<float, 1> m_nda_shist;

    //float*    p_rsum;
    //unsigned* p_rsta;
    //float*    p_ssta;
    float     p_rsum[2000];
    unsigned  p_rsta[2000];
    float     p_ssta[2000];
};

//--------------------------
} // namespace PSQt

#endif // IMAGEPROC_H
//--------------------------
