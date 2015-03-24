#ifndef LABCOLORRING_H
#define LABCOLORRING_H

#include <QWidget>
#include <QLabel>
#include <QPainter>
#include <QPixmap>
#include <QPoint>

#include <Qt>
#include <QtGui>
#include <QtCore>

namespace PSQt {


  //class WdgColorTable;

class LabColorRing : public QLabel
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:

  static const float RAD2DEG = 180/3.14159265;
  static const float DEG2RAD = 3.14159265/180;

  LabColorRing(QWidget* parent, const unsigned& ssize, float &h1, float &h2 );

    void setFrame() ;
    void setStyle() ;
    void setPens() ;
    void showTips() ;

    void resizeEvent      (QResizeEvent *event = 0) ;
    void closeEvent       (QCloseEvent  *event = 0) ;
    void moveEvent        (QMoveEvent   *event = 0) ;

    void enterEvent       (QEvent       *event = 0) ;
    void leaveEvent       (QEvent       *event = 0) ;

    void paintEvent       (QPaintEvent  *event = 0) ;
    void setHueAngle      (QMouseEvent  *event = 0) ;

    void setColorRing(const int& ssize=512) ;
    void drawLines();
    void drawCircs();
    void setPoints();

 protected :

    void mousePressEvent  (QMouseEvent  *event = 0) ;
    void mouseMoveEvent   (QMouseEvent  *event = 0) ;
    void mouseReleaseEvent(QMouseEvent  *event = 0) ;

    //bool eventFilter     (QObject *obj, QEvent *event = 0) ;



 public slots:
    void onButExit() ;
    void onSetShifter(const unsigned& selected) ;

 signals :
    void hueAngleIsMoving(const unsigned& selected) ;
    void hueAngleIsMoved() ;

 private:

    QWidget*     m_parent;
    int          m_ssize; 
    float        m_R; 
    float        m_frR1; 
    float        m_frR2; 
    float        m_R1; 
    float        m_R2; 

    float&       m_h1; 
    float&       m_h2; 

    float        m_ang; 
    float        m_n360;
    float        m_ang_old; 

    QPointF      m_poiC;
    QPointF      m_poi1;
    QPointF      m_poi2;
    QPointF      m_poi1e;
    QPointF      m_poi2e;
    QPainter     m_painter;
    QPen*        m_pen_w1;
    QPen*        m_pen_w3;
    QPen*        m_pen1;
    QPen*        m_pen2;

    float        m_rpicker; 
    unsigned     m_selected; 

    QFrame*      m_frame;
    QPixmap*     m_pixmap_cring;
    QGraphicsScene* m_scene;
};

} // namespace PSQt

#endif // LABCOLORRING_H
