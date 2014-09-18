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

//class WdgImage : public QWidget
class WdgImage : public QLabel
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
  WdgImage( QWidget *parent = 0
          , const std::string& fname_geo=std::string()
          , const std::string& fname_img=std::string()
          ); 

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
    void setColorPixmap() ;
    void setColorWhellPixmap() ;
    void setCameraImage() ;
    void drawLine() ;
    void drawRect() ;
    void zoomInImage() ;
    void setPixmapScailedImage(const QImage* = 0);
    void setColorBar(const unsigned& rows =   20, 
                     const unsigned& cols = 1024,
                     const float&    hue1 = -120,
                     const float&    hue2 =   60);

 private:
    //QLabel*      m_lab_fname;
    QFrame*      m_frame;
    QPainter*    m_painter;
    QScrollArea* m_scroll_area;

    GeoImage* m_geo_img;

    QPixmap* m_pixmap_raw;
    QPixmap* m_pixmap_scl;

    QPen*   m_pen1;
    QPen*   m_pen2;
    QPoint* m_point1;
    QPoint* m_point2;
    QRect*  m_rect1;
    QRect*  m_rect2;
    bool    m_is_pushed;
};

} // namespace PSQt

#endif // WDGIMAGE_H
