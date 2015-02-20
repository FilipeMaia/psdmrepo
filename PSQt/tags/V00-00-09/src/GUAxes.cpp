//--------------------------
#include "PSQt/GUAxes.h"
//#include "PSQt/QGUtils.h"
#include <PSQt/GURuler.h>

#include <QCloseEvent>
#include <QResizeEvent>
#include <QMouseEvent>


#include <iostream>    // for std::cout
//#include <fstream>     // for std::ifstream(fname)
//#include <math.h>  // atan2
//#include <cstring> // for memcpy

//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

  GUAxes::GUAxes(  QWidget *parent
                   , const float& xmin
		   , const float& xmax
		   , const float& ymin
		   , const float& ymax
                   , const unsigned pbits
                 ) 
  : QGraphicsView(parent)
  , m_xmin(xmin)
  , m_xmax(xmax)
  , m_ymin(ymin)
  , m_ymax(ymax)
  , m_pbits(pbits)
{
  if(m_pbits & 1) {std::cout << "GUAxes - ctor\n"; printMemberData();}

  pview()->setGeometry(100, 50, 700, 400);
  pview()->setMinimumSize(300,200);
 
  QRectF sceneRect(xmin, ymin, xmax-xmin, ymax-ymin); 
  m_scene = new QGraphicsScene(sceneRect); 

  pview()->setScene(m_scene);
  pview()->ensureVisible(sceneRect, 100, 100);
  //this->fitInView(sceneRect, Qt::IgnoreAspectRatio);

  this->updateTransform();
  this->setAxes();

  if(m_pbits & 2) this->printTransform();
}

//--------------------------

  void GUAxes::updateTransform()
  {
    const QRect & geometryRect = pview()->geometry();

    //std::cout << "geometryRect.width() :" << geometryRect.width() << '\n';
    //std::cout << "geometryRect.height():" << geometryRect.height() << '\n';

    float sx =  (geometryRect.width()-100)/(m_xmax-m_xmin);
    float sy = -(geometryRect.height()-100)/(m_ymax-m_ymin);
    QTransform trans(sx, 0, 0, sy, 20, 20);

    pview()->setTransform(trans); // The transformation matrix tranforms the scene into view coordinates.
  }

//--------------------------

  void GUAxes::setAxes()
  {
    QPen pen1(Qt::black, 2, Qt::SolidLine);
    pen1.setCosmetic(true);

    //QGraphicsScene * scene = pview()->scene();
    
    PSQt::GURuler rulerhd(rview(),"HD",m_xmin,m_xmax,m_ymin);
    pscene()->addPath(rulerhd.pathForRuler(), pen1);

    PSQt::GURuler rulerhu(rview(),"HU",m_xmin,m_xmax,m_ymax,5,2,0,0,0);
    pscene()->addPath(rulerhu.pathForRuler(), pen1);
    
    PSQt::GURuler rulervl(rview(),"VL",m_ymin,m_ymax,m_xmin,4,2);
    pscene()->addPath(rulervl.pathForRuler(), pen1);
    
    PSQt::GURuler rulervr(rview(),"VR",m_ymin,m_ymax,m_xmax,4,2,0,0,0);
    pscene()->addPath(rulervr.pathForRuler(), pen1);
  }

//--------------------------
  void GUAxes::printMemberData()
  {
    std::cout << "GUAxes::printMemberData():"
              << "\n xmin:" << m_xmin
              << "\n xmax:" << m_xmax
              << "\n ymin:" << m_ymin
              << "\n ymax:" << m_ymax
              << "\n pbits:"<< m_pbits
              << '\n';
  }

//--------------------------

  void GUAxes::printTransform()
  {
    QTransform trans = pview()->transform();
    std::cout << "GUAxes::printTransform():"
              << "\n m11():" << trans.m11()
    	      << "\n m11():" << trans.m11()
    	      << "\n dx() :" << trans.dx()
    	      << "\n dy() :" << trans.dy()
              << '\n';
  }

//--------------------------

void 
GUAxes::closeEvent(QCloseEvent *event)
{
  //QGraphicsView::closeEvent(event);
  std::cout << "GUAxes::closeEvent(...): type = " << event -> type() << std::endl;
}

//--------------------------

void 
GUAxes::resizeEvent(QResizeEvent *event)
{
  //m_frame -> setFrameRect (this->rect());
  //m_frame->setGeometry(0, 0, event->size().width(), event->size().height());
  
  //std::cout << "GUAxes::resizeEvent(...): w=" << event->size().width() 
  //          << "  h=" << event->size().height() << '\n';

  //setWindowTitle("Window is resized");

  this->updateTransform();

}

//--------------------------

void 
GUAxes::mousePressEvent(QMouseEvent *e)
{
  /*
  std::cout << "mousePressEvent:"
            << "  button: " << e->button()
            << "  x(),y() = " << e->x() << ", " << e->y()
            << "  isActiveWindow(): " << this->isActiveWindow()
            << '\n';
  */
  //m_point1->setX(e->x());
  //m_point1->setY(e->y());
  //m_point2->setX(e->x());
  //m_point2->setY(e->y());

  //m_is_pushed = true;
  //update();
}

//--------------------------

void 
GUAxes::mouseReleaseEvent(QMouseEvent *e)
{
  /*
  std::cout << "mouseReleaseEvent:"
            << "  button: " << e->button()
            << "  x(),y() = " << e->x() << ", " << e->y()
            << '\n';
  */
  /*
  m_is_pushed = false;

  if(e->button() == 4) { // for middle button
    setPixmapScailedImage();
    update();
    return;
  }

  m_point2->setX(e->x());
  m_point2->setY(e->y());

  QPoint dist = *m_point2 - *m_point1;
  if(this->rect().contains(*m_point2) && dist.manhattanLength() > 5) zoomInImage();

  update();
  */
}

//--------------------------

void 
GUAxes::mouseMoveEvent(QMouseEvent *e)
{
  //std::cout << "mouseMoveEvent: "
  //          << "  x(),y() = "  << e->x() << ", " << e->y()
  //          << '\n';
  //m_point2->setX(e->x());
  //m_point2->setY(e->y());
  //update();
}


//--------------------------


//--------------------------
//--------------------------
//--------------------------
//--------------------------
} // namespace PSQt
//--------------------------
