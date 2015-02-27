#ifndef PSQT_WDGIMAGEFIGS_H
#define PSQT_WDGIMAGEFIGS_H

#include "PSQt/WdgImage.h"
#include "PSQt/DragStore.h"

/*
#include <QFrame>
#include <QPainter>
#include <QPen>
#include <QPoint>
#include <QRect>
*/

#include <QCloseEvent>
#include <QResizeEvent>
#include <QMoveEvent>
#include <QMouseEvent>

#include <QtCore>
//#include <QtGui>
//#include <Qt>


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

//class WdgImageFigs : public QWidget
//class WdgImageFigs : public QLabel
class WdgImageFigs : public WdgImage
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
    WdgImageFigs(QWidget *parent = 0, const std::string& fname=std::string()); 
    WdgImageFigs(QWidget *parent, const QImage* image);
    virtual ~WdgImageFigs();

    inline WdgImage* wdgImage(){ return (WdgImage*) this; }
    //inline WdgImage* wdgImage(){ return WdgImage::getThis(); }
    const QPointF& getCenter(); //{ return m_dragstore->getCenter(); }

    void addCircle(const float& rad_raw=100);

 public slots:
    void onTest() ;

 protected:
    void setFrame() ;
    void paintEvent(QPaintEvent *event = 0) ;
    void closeEvent(QCloseEvent *event = 0) ;
    void moveEvent(QMoveEvent *event = 0) ;
    void resizeEvent(QResizeEvent *event = 0) ;
    void mousePressEvent(QMouseEvent *event = 0) ;
    void mouseReleaseEvent(QMouseEvent *event = 0) ;
    void mouseMoveEvent(QMouseEvent *event = 0) ;

 private:
    QPainter*    m_painter;
    DragStore*   m_dragstore;
    DRAGMODE     m_dragmode;


    /*
    GeoImage*    m_geo_img; 

    QPixmap*     m_pixmap_raw;
    QPixmap*     m_pixmap_scl;
    QPoint*      m_point1;
    QPoint*      m_point2;
    QRect*       m_rect1;
    QRect*       m_rect2;
    */

    QPen*        m_pen1;
    QPen*        m_pen2;

    inline const char* _name_(){return "WdgImageFigs";}
    void setParameters();

    void drawDragFigs() ;
    void drawDragFigsV0() ;
    void drawLine() ;
    void drawRect() ;
    //void drawCirc() ;
    //void drawCenter() ;
};

} // namespace PSQt

#endif // PSQT_WDGIMAGEFIGS_H
