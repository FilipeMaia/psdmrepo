#ifndef GUAXES_H
#define GUAXES_H

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

  //enum ORIDIR {HORUP, HORDOWN, VERRIGHT, VERLEFT};

  GUAxes( QWidget *parent=0
	  , const float& xmin =  0
	  , const float& xmax = 30
	  , const float& ymin =-10
	  , const float& ymax = 20 
	  , const unsigned pbits = 0177777); 

  virtual ~GUAxes () {}

  void updateTransform();
  void setAxes();
  void printMemberData();
  void printTransform();

  QTransform  transform() { return pview()->transform(); }
  QGraphicsScene * pscene() { return pview()->scene(); }
  //QGraphicsView *  pview() { return dynamic_cast<QGraphicsView*>(this); }
  //QGraphicsView &  rview() { return *dynamic_cast<QGraphicsView*>(this); }
  QGraphicsView *  pview() { return this; }
  QGraphicsView &  rview() { return *this; }

 protected:
    void closeEvent(QCloseEvent *event = 0) ;
    void resizeEvent(QResizeEvent *event = 0) ;
    void mousePressEvent(QMouseEvent *event = 0) ;
    void mouseReleaseEvent(QMouseEvent *event = 0) ;
    void mouseMoveEvent(QMouseEvent *event = 0) ;

private:

  //QGraphicsView& m_view;
    std::string    m_oristr;
    float          m_xmin;
    float          m_xmax;
    float          m_ymin;
    float          m_ymax;
    unsigned       m_pbits;

    QGraphicsScene* m_scene; 

    //QPainterPath m_path;
};

//--------------------------
} // namespace PSQt

#endif // GUAXES_H
