//--------------------------
#include "PSQt/GUAxes.h"
//#include "PSQt/QGUtils.h"
#include "PSQt/Logger.h"

#include <QCloseEvent>
#include <QResizeEvent>
#include <QMouseEvent>

#include <sstream>  // for stringstream
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
		   , const unsigned& nxdiv1
		   , const unsigned& nydiv1
		   , const unsigned& nxdiv2
		   , const unsigned& nydiv2
                   , const unsigned pbits
                 ) 
  : QGraphicsView(parent)
  , m_pbits(pbits)
  , m_color(Qt::black)
  , m_pen(Qt::black, 2, Qt::SolidLine)
  , m_font()
  , m_ruler_hd(0)
  , m_ruler_hu(0)
  , m_ruler_vl(0)
  , m_ruler_vr(0)
{
  m_pen.setCosmetic(true);

  if(m_pbits & 1) {std::cout << "GUAxes - ctor\n"; printMemberData();}

  pview()->setGeometry(100, 50, 700, 400);
  pview()->setMinimumSize(300,200);
  //this->setContentsMargins(-9,-9,-9,-9);
 
  m_scene = new QGraphicsScene(); 
  pview()->setScene(m_scene);
  
  setLimits(xmin, xmax, ymin, ymax, nxdiv1, nydiv1, nxdiv2, nydiv2);

  if(m_pbits & 2) this->printTransform();

  connectForTest(); 
}

//--------------------------

  void GUAxes::connectForTest() 
  {
    // connections for internal test 
    connect(this, SIGNAL(pressOnAxes(QMouseEvent*, QPointF)),
            this, SLOT(testSignalPressOnAxes(QMouseEvent*, QPointF)));

    connect(this, SIGNAL(releaseOnAxes(QMouseEvent*, QPointF)),
            this, SLOT(testSignalReleaseOnAxes(QMouseEvent*, QPointF)));

    //connect(this, SIGNAL(moveOnAxes(QMouseEvent*, QPointF)),
    //        this, SLOT(testSignalMoveOnAxes(QMouseEvent*, QPointF)));

  }

//--------------------------

  void GUAxes::setLimits( const float& xmin
		        , const float& xmax
		        , const float& ymin
		        , const float& ymax
		        , const unsigned& nxdiv1
                        , const unsigned& nydiv1
		        , const unsigned& nxdiv2
                        , const unsigned& nydiv2 )
  {
    m_xmin = xmin;
    m_xmax = xmax;
    m_ymin = ymin;
    m_ymax = ymax;
    m_nxdiv1 = nxdiv1;
    m_nydiv1 = nydiv1;
    m_nxdiv2 = nxdiv2;
    m_nydiv2 = nydiv2;

    this->updateTransform();
    this->setAxes();
  }

//--------------------------
  GUAxes::~GUAxes () 
  {
    if (m_ruler_hd) delete m_ruler_hd;
    if (m_ruler_hu) delete m_ruler_hu;
    if (m_ruler_vl) delete m_ruler_vl;
    if (m_ruler_vr) delete m_ruler_vr;
  }

//--------------------------

  void GUAxes::updateTransform()
  {
    const QRect & geometryRect = pview()->geometry();

    //std::stringstream ss; ss << "h:" << geometryRect.width()  << "  w:" << geometryRect.height();
    //setWindowTitle(ss.str().c_str());
    //std::cout << ss.str() << '\n';

    float sx =  (geometryRect.width()-150)/(m_xmax-m_xmin);
    float sy = -(geometryRect.height()-50)/(m_ymax-m_ymin);
    QTransform trans(sx, 0, 0, sy, 0, 0);

    pview()->setTransform(trans); // The transformation matrix tranforms the scene into view coordinates.

    //this->printTransform();
  }

//--------------------------

  void GUAxes::setAxes()
  {
    m_scene_rect.setRect(m_xmin, m_ymin, m_xmax-m_xmin, m_ymax-m_ymin); 

    m_scene->setSceneRect(m_scene_rect);
    //pview()->fitInView(m_scene_rect, Qt::IgnoreAspectRatio);
    //pview()->ensureVisible(m_scene_rect, 100, 100);
    //pview()->ensureVisible(m_scene_rect);

    //QGraphicsScene * scene = pview()->scene();

    if (m_ruler_hd) delete m_ruler_hd;
    if (m_ruler_hu) delete m_ruler_hu;
    if (m_ruler_vl) delete m_ruler_vl;
    if (m_ruler_vr) delete m_ruler_vr;

    m_ruler_hd = new PSQt::GURuler(rview(),GURuler::HD,m_xmin,m_xmax,m_ymin,m_nxdiv1,m_nxdiv2,1,0,0,0,0,m_color,m_pen,m_font);
    m_ruler_hu = new PSQt::GURuler(rview(),GURuler::HU,m_xmin,m_xmax,m_ymax,m_nxdiv1,m_nxdiv2,0,0,0,0,0,m_color,m_pen,m_font);
    m_ruler_vl = new PSQt::GURuler(rview(),GURuler::VL,m_ymin,m_ymax,m_xmin,m_nydiv1,m_nydiv2,1,0,0,0,0,m_color,m_pen,m_font);
    m_ruler_vr = new PSQt::GURuler(rview(),GURuler::VR,m_ymin,m_ymax,m_xmax,m_nydiv1,m_nydiv2,0,0,0,0,0,m_color,m_pen,m_font);
  }

//--------------------------
  void GUAxes::printMemberData()
  {
    std::cout << "GUAxes::printMemberData():"
              << "\n xmin  :" << m_xmin
              << "\n xmax  :" << m_xmax
              << "\n ymin  :" << m_ymin
              << "\n ymax  :" << m_ymax
              << "\n nxdiv1:" << m_nxdiv1
              << "\n nydiv1:" << m_nydiv1
              << "\n nxdiv2:" << m_nxdiv2
              << "\n nydiv2:" << m_nydiv2
              << "\n pbits :" << m_pbits
              << '\n';
  }

//--------------------------

  void GUAxes::printTransform()
  {
    QTransform trans = pview()->transform();
    std::cout << "GUAxes::printTransform():"
              << "\n m11():" << trans.m11()
    	      << "\n m22():" << trans.m22()
    	      << "\n dx() :" << trans.dx()
    	      << "\n dy() :" << trans.dy()
              << '\n';
  }

//--------------------------

void 
GUAxes::closeEvent(QCloseEvent *event)
{
  //QGraphicsView::closeEvent(event);
  //std::cout << "GUAxes::closeEvent(...): type = " << event -> type() << std::endl;
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



  //pview()->fitInView(m_scene_rect, Qt::IgnoreAspectRatio);
  this->updateTransform();
}

//--------------------------

QPointF 
GUAxes::pointOnAxes(QMouseEvent *e)
{
  //const QPoint& poswin = e->pos();
  QPointF sp = pview()->mapToScene(e->pos());
  return sp;
}

//--------------------------

void 
GUAxes::mousePressEvent(QMouseEvent *e)
{
  QPointF sp = pointOnAxes(e);
  emit pressOnAxes(e, sp);
}

//--------------------------

void 
GUAxes::mouseReleaseEvent(QMouseEvent *e)
{
  QPointF sp = pointOnAxes(e);
  emit releaseOnAxes(e, sp);
}

//--------------------------

void 
GUAxes::mouseMoveEvent(QMouseEvent *e)
{
  //QPointF sp = pointOnAxes(e);
  //emit moveOnAxes(e, sp);
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------

void
GUAxes::message(QMouseEvent *e, const QPointF& sp, const char* cmt)
{
  std::stringstream ss;
  ss << _name_() << " " << cmt
     << "  button: " << e->button()
     << "  window x(), y() = " << e->x() << ", " << e->y()
     << "  scene x(), y() = " << sp.x() << ", " << sp.y();

  MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------

void 
GUAxes::testSignalPressOnAxes(QMouseEvent* e, QPointF p)
{
  message(e, p, "press   ");
}

//--------------------------

void 
GUAxes::testSignalReleaseOnAxes(QMouseEvent* e, QPointF p)
{
  message(e, p, "release");
}

//--------------------------

void 
GUAxes::testSignalMoveOnAxes(QMouseEvent* e, QPointF p)
{
  message(e, p, "move    ");
}

//--------------------------
//--------------------------
//--------------------------
} // namespace PSQt
//--------------------------
