#ifndef IMAGEPROC_H
#define IMAGEPROC_H
//--------------------------

#include "PSQt/GeoImage.h"
#include "ndarray/ndarray.h"
#include <stdint.h> // uint8_t, uint32_t, etc.

#include <QObject>
#include <QPointF>
 
namespace PSQt {

//--------------------------

/**
 *  @ingroup PSQt
 *
 *  @brief ImageProc - processing on raw image ndarray using zoom rect selection
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

class ImageProc : public QObject
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:

  typedef PSQt::GeoImage::raw_image_t raw_image_t; // double

  ImageProc(); 
  virtual ~ImageProc();

    //const ndarray<const raw_image_t, 2> getImage();

 public slots:
    void onImageIsUpdated(const ndarray<const GeoImage::raw_image_t,2>&);
    void onZoomIsChanged(int&, int&, int&, int&);
    void onCenterIsChanged(const QPointF&);
    //void onCenterIsMoved(const QPointF&); // signal on each motion of coursor
    void testSignalRHistIsFilled(ndarray<float, 1>&);

 signals :
    void rhistIsFilled(ndarray<float, 1>&);

 private:
    inline const char* _name_(){return "ImageProc";}
    bool evaluateRadialIndexes();
    bool fillRadialHistogram();
    bool fillSpectralHistogram();

    float m_rbin_width;
    bool m_image_is_set;
    bool m_center_is_set;
    bool m_zoom_is_set;
    bool m_rindx_is_set;
    bool m_rhist_is_set;

    unsigned m_ixmax;
    unsigned m_iymax;
    unsigned m_irmax;

    int m_zxmin;
    int m_zymin;
    int m_zxmax;
    int m_zymax;

    QPointF m_center;
    ndarray<GeoImage::raw_image_t,2> m_nda_image; 
    ndarray<unsigned,2> m_nda_rindx;
    ndarray<float, 1> m_nda_rhist;

    float*    p_rsum;
    unsigned* p_rsta;
};

//--------------------------
} // namespace PSQt

#endif // IMAGEPROC_H
//--------------------------
