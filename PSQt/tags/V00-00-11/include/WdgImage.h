#ifndef PSQT_WDGIMAGE_H
#define PSQT_WDGIMAGE_H

#include "PSQt/GeoImage.h"

#include <QWidget>
#include <QLabel>
#include <QFrame>
#include <QPainter>
#include <QPen>

#include <QPoint>
#include <QRect>

#include <QPixmap>
#include <QImage>
#include <QBitmap>

#include <QCloseEvent>
#include <QResizeEvent>
#include <QMouseEvent>

//#include <QScrollBar>
//#include <QScrollArea>

//#include <Qt>
//#include <QtGui>
#include <QtCore>


namespace PSQt {

// /// @addtogroup PSQt PSQt

/**
 *  @ingroup PSQt
 * 
 *  @brief Shows image in the QLabel box.
 * 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see GUView
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

//class WdgImage : public QWidget
class WdgImage : public QLabel
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
  typedef GeoImage::image_t image_t;


  WdgImage(QWidget *parent = 0, const std::string& fname=std::string()); 
  WdgImage(QWidget *parent, const QImage* image);
  virtual ~WdgImage();

  inline QPainter* getPainter(){ return m_painter; }
  //inline WdgImage* getThis(){ return this; }

  QPointF pointInImage(const QPointF& point_raw);
  QPointF pointInRaw(const QPointF& point_raw);

 public slots:
    void onFileNameChanged(const std::string& fname) ;
    void onTest() ;
    void onImageIsUpdated(const ndarray<const GeoImage::raw_image_t,2>&) ;
    void onNormImageIsUpdated(const ndarray<GeoImage::image_t,2>&) ;

 protected:

    int          m_xmin_raw;
    int          m_xmax_raw; 
    int          m_ymin_raw;
    int          m_ymax_raw;

    QPixmap*     m_pixmap_raw;
    QPixmap*     m_pixmap_scl;

    void setFrame() ;
    void paintEvent(QPaintEvent *event = 0) ;
    void closeEvent(QCloseEvent *event = 0) ;
    void resizeEvent(QResizeEvent *event = 0) ;
    void mousePressEvent(QMouseEvent *event = 0) ;
    void mouseReleaseEvent(QMouseEvent *event = 0) ;
    void mouseMoveEvent(QMouseEvent *event = 0) ;
    void loadImageFromFile(const std::string& fname=std::string()) ;

    void drawRect() ;
    void zoomInImage() ;
    void setPixmapScailedImage(const QImage* = 0) ;

    void setCameraImage( const std::string& fname_geo=std::string()
                       , const std::string& fname_img=std::string()) ;
    void setColorPixmap() ;
    void setColorWhellPixmap() ;
    void setColorBar(const unsigned& rows =   20, 
                     const unsigned& cols = 1024,
                     const float&    hue1 = -120,
                     const float&    hue2 =   60) ;

 private:
    QFrame*      m_frame;
    QPainter*    m_painter;

    GeoImage*    m_geo_img;

    QPen*        m_pen1;
    QPen*        m_pen2;
    QPoint*      m_point1;
    QPoint*      m_point2;
    QRect*       m_rect1;
    QRect*       m_rect2;
    bool         m_is_pushed;
    bool         m_zoom_is_on;

    unsigned     m_ncolors;
    float        m_hue1;
    float        m_hue2;

    inline const char* _name_(){return "WdgImage";}
    void setWdgParams() ;
    void resetZoom() ;
};

} // namespace PSQt

#endif // PSQT_WDGIMAGE_H
