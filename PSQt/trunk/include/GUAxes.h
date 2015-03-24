#ifndef GUAXES_H
#define GUAXES_H

#include <PSQt/GURuler.h>

#include <string>
#include <QWidget>
#include <QGraphicsView>
#include <QTransform>
#include <QGraphicsScene>

//#include <QPoint>
//#include <QPainterPath>
//#include <QPen>
//#include <QBrash>

namespace PSQt {

//--------------------------

class GUAxes : public QGraphicsView
{
  Q_OBJECT // macro is needed for connection of signals and slots

public:

  GUAxes( QWidget *parent=0
	  , const float& xmin      = 0
	  , const float& xmax      = 100
	  , const float& ymin      = 0
	  , const float& ymax      = 100 
	  , const unsigned& nxdiv1 = 4
	  , const unsigned& nydiv1 = 4
	  , const unsigned& nxdiv2 = 2
	  , const unsigned& nydiv2 = 2
	  , const unsigned pbits   = 0); // 0177777

  virtual ~GUAxes ();

  void setLimits( const float& xmin = 0
       	        , const float& xmax = 100
       	        , const float& ymin = 0
       	        , const float& ymax = 100
       	        , const unsigned& nxdiv1 = 4
                , const unsigned& nydiv1 = 4
       	        , const unsigned& nxdiv2 = 2
		, const unsigned& nydiv2 = 2);

  void updateTransform();
  void setAxes();
  void setPen(const QPen& pen) { m_pen=pen; }
  void setFont(const QFont& font) { m_font=font; }
  void setColor(const QColor& color) { m_color=color; }
  void printMemberData();
  void printTransform();

  QTransform  transform()  { return pview()->transform(); }
  QGraphicsScene * pscene(){ return pview()->scene(); }
  QGraphicsView *  pview() { return this; }
  QGraphicsView &  rview() { return *this; }


 signals :
    void pressOnAxes(QMouseEvent*, QPointF) ;
    void releaseOnAxes(QMouseEvent*, QPointF) ;
    void moveOnAxes(QMouseEvent*, QPointF) ;

 public slots:
    void testSignalPressOnAxes(QMouseEvent*, QPointF) ;
    void testSignalReleaseOnAxes(QMouseEvent*, QPointF) ;
    void testSignalMoveOnAxes(QMouseEvent*, QPointF) ;

 protected:
    void closeEvent(QCloseEvent *event = 0) ;
    void resizeEvent(QResizeEvent *event = 0) ;
    void mousePressEvent(QMouseEvent *event = 0) ;
    void mouseReleaseEvent(QMouseEvent *event = 0) ;
    void mouseMoveEvent(QMouseEvent *event = 0) ;

private:

  //QGraphicsView&  m_view;
    float           m_xmin;
    float           m_xmax;
    float           m_ymin;
    float           m_ymax;
    unsigned        m_nxdiv1;
    unsigned        m_nydiv1;
    unsigned        m_nxdiv2;
    unsigned        m_nydiv2;
    unsigned        m_pbits;
    QColor          m_color;
    QPen            m_pen;
    QFont           m_font;

    QGraphicsScene* m_scene; 
    QRectF          m_scene_rect;

    PSQt::GURuler* m_ruler_hd; 
    PSQt::GURuler* m_ruler_hu; 
    PSQt::GURuler* m_ruler_vl; 
    PSQt::GURuler* m_ruler_vr; 

    inline const char* _name_(){return "GUAxes";}
    QPointF pointOnAxes(QMouseEvent *e);
    void connectForTest(); 
    void message(QMouseEvent *e, const QPointF& sp, const char* cmt="");
};

//--------------------------
} // namespace PSQt

#endif // GUAXES_H
