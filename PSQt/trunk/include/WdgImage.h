#ifndef WDGIMAGE_H
#define WDGIMAGE_H

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
#include <QScrollArea>

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
 *  @version $Id:$
 *
 *  @author Mikhail Dubrovin
 */

//class WdgImage : public QWidget
class WdgImage : public QLabel
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
  WdgImage(QWidget *parent = 0, const std::string& fname=std::string()); 
  WdgImage(QWidget *parent, const QImage* image);

 public slots:
    void onFileNameChanged(const std::string& fname) ;
    void onTest() ;

 protected:
    void setFrame() ;
    void paintEvent(QPaintEvent *event = 0) ;
    void closeEvent(QCloseEvent *event = 0) ;
    void resizeEvent(QResizeEvent *event = 0) ;
    void mousePressEvent(QMouseEvent *event = 0) ;
    void mouseReleaseEvent(QMouseEvent *event = 0) ;
    void mouseMoveEvent(QMouseEvent *event = 0) ;
    void loadImageFromFile(const std::string& fname=std::string()) ;

    void drawLine() ;
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
    QScrollArea* m_scroll_area;

    GeoImage*    m_geo_img;

    QPixmap*     m_pixmap_raw;
    QPixmap*     m_pixmap_scl;

    QPen*        m_pen1;
    QPen*        m_pen2;
    QPoint*      m_point1;
    QPoint*      m_point2;
    QRect*       m_rect1;
    QRect*       m_rect2;
    bool         m_is_pushed;


    void setWdgParams() ;
};

} // namespace PSQt

#endif // WDGIMAGE_H
