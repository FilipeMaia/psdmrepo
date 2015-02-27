//--------------------------
#include "PSQt/WdgImageFigs.h"
#include "PSQt/Logger.h"
//#include "PSCalib/GeometryAccess.h"
//#include "ndarray/ndarray.h" // for img_from_pixel_arrays(...)

#include "PSQt/QGUtils.h"
#include "PSQt/DragStore.h"

#include <iostream>    // for std::cout
#include <fstream>     // for std::ifstream(fname)
//using namespace std; // for cout without std::
//#include <math.h>  // atan2
//#include <cstring> // for memcpy

namespace PSQt {

//--------------------------

WdgImageFigs::WdgImageFigs(QWidget *parent, const std::string& ifname)
  : WdgImage(parent, ifname)
{
  this->setParameters();
}

//--------------------------

WdgImageFigs::WdgImageFigs( QWidget *parent, const QImage* image)
  : WdgImage(parent, image)
{
  this->setParameters();
}

//--------------------------

WdgImageFigs::~WdgImageFigs()
{
}

//--------------------------

void
WdgImageFigs::setParameters()
{
  //m_dragstore = new DragStore(WdgImage::getThis());
  m_dragstore = new DragStore(this);
  m_dragmode = ZOOM;

  QVector<qreal> dashes;
  qreal space = 5;
  dashes << 5 << space;

  m_pen1 = new QPen(Qt::red,   2, Qt::DashLine);
  m_pen2 = new QPen(Qt::green, 2, Qt::DashLine);
  m_pen1->setDashPattern(dashes);
  m_pen2->setDashPattern(dashes);
  m_pen2->setDashOffset(5);
}

//--------------------------

void 
WdgImageFigs::paintEvent(QPaintEvent *e)
{
  WdgImage::QLabel::paintEvent(e);
  WdgImage::paintEvent(e);

  m_painter = WdgImage::getPainter();
  m_painter->begin(this);

  //-----------
  drawDragFigs();
  //drawDragFigsV0();
  //-----------

  m_painter->end();

  //static unsigned count=0; count++;
  //std::cout << "WdgImageFigs::paintEvent counter = " << count << '\n';
}

//--------------------------

void 
WdgImageFigs::drawDragFigs()
{
  m_dragstore -> drawFigs(m_dragmode);
}

//--------------------------

void 
WdgImageFigs::addCircle(const float& rad_raw)
{
  m_dragstore -> addCircle(rad_raw);
  update();
}
//--------------------------

const QPointF& 
WdgImageFigs::getCenter()
{ 
  return m_dragstore->getCenter(); 
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------

void 
WdgImageFigs::drawDragFigsV0()
{
  drawLine();
  drawRect();
  //drawCirc();
  //drawCenter();
}

//--------------------------

void 
WdgImageFigs::drawLine()
{
  QPointF p1_raw(200,100);
  QPointF p2_raw(800,800);

  QPointF p1 = WdgImage::pointInImage(p1_raw);
  QPointF p2 = WdgImage::pointInImage(p2_raw);

  QPen pen(Qt::white, 2, Qt::SolidLine);
  m_painter->setPen(pen);
  m_painter->drawLine(p1, p2);
}

//--------------------------

void 
WdgImageFigs::drawRect()
{
  QPointF p1_raw(100,200);
  QPointF p2_raw(200,500);

  QPointF p1 = WdgImage::pointInImage(p1_raw);
  QPointF p2 = WdgImage::pointInImage(p2_raw);

  QRectF m_rect(p1,p2);
  m_painter->setPen(*m_pen1); m_painter->drawRect(m_rect); 
  m_painter->setPen(*m_pen2); m_painter->drawRect(m_rect); 
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------

void 
WdgImageFigs::moveEvent(QMoveEvent *e)
{
  WdgImage::moveEvent(e);
  //stringstream ss; 
  //ss << _name_() << "::moveEvent(...): x=" << e->pos().x() << "  y=" << e->pos().y();
  //MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------

void 
WdgImageFigs::resizeEvent(QResizeEvent *e)
{
  WdgImage::resizeEvent(e);
  //stringstream ss; 
  //ss << _name_() << "::resizeEvent(...): w=" << e->size().width() << "  h=" << e->size().height();
  //MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------

void 
WdgImageFigs::closeEvent(QCloseEvent *e)
{
  QWidget::closeEvent(e);
  //stringstream ss; ss << _name_() << "::closeEvent(...): type = " << e -> type();
  //std::cout << ss.str() << '\n';
  //MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------

void 
WdgImageFigs::mousePressEvent(QMouseEvent *e)
{
  bool click_on_fig = m_dragstore -> containFigs(e->posF());

  if(click_on_fig) {
      if(e->button() == Qt::MidButton) m_dragmode = DELETE;
      else m_dragmode = MOVE;
  }

  if(m_dragmode == ZOOM) WdgImage::mousePressEvent(e);

  if(m_dragmode == MOVE)   MsgInLog(_name_(), DEBUG, "MOVE mode is on");

  if(m_dragmode == DELETE) MsgInLog(_name_(), DEBUG, "DELETE mode is on");
  //std::cout << _name_() << "::mousePressEvent:"
  //          << "  button: " << e->button()
  //          << "  x(),y() = " << e->x() << ", " << e->y()
  //          << '\n';

  /*
  QPointF pimg(float(e->x()),float(e->y())); 
  QPointF praw  = pointInRaw(pimg);
  QPointF pimg1 = pointInImage(praw);
  std::cout << "  point in image   x:" << pimg.x()  << "   x:" << pimg.y() << '\n';
  std::cout << "  point in raw     x:" << praw.x()  << "   x:" << praw.y() << '\n';
  std::cout << "  point in image1  x:" << pimg1.x() << "   x:" << pimg1.y() << '\n';
  */

  this -> setCursor(Qt::ClosedHandCursor);
  /*
  m_point1->setX(e->x());
  m_point1->setY(e->y());
  m_point2->setX(e->x());
  m_point2->setY(e->y());
  */
  update();
}

//--------------------------

void 
WdgImageFigs::mouseReleaseEvent(QMouseEvent *e)
{
  if(m_dragmode == ZOOM) WdgImage::mouseReleaseEvent(e);

  if(m_dragmode == DELETE) m_dragstore -> deleteFig();

  if(m_dragmode == MOVE) m_dragstore -> moveFigsIsCompleted(e->posF());

  //std::cout << _name_() << "::mouseReleaseEvent:"
  //          << "  button: " << e->button()
  //          << "  x(),y() = " << e->x() << ", " << e->y()
  //          << '\n';

  //this -> unsetCursor(); 
  this -> setCursor(Qt::PointingHandCursor); // Qt::SizeAllCursor, Qt::WaitCursor, Qt::PointingHandCursor

  //if(e->button() == 4) { // for middle button
  //  update();
  //  return;
  //}

  //m_point2->setX(e->x());
  //m_point2->setY(e->y());

  update();
  m_dragmode = ZOOM;
}

//--------------------------

void 
WdgImageFigs::mouseMoveEvent(QMouseEvent *e)
{
  if(m_dragmode == ZOOM) WdgImage::mouseMoveEvent(e);

  if(m_dragmode == MOVE) m_dragstore -> moveFigs(e->posF());

  //std::cout << "mouseMoveEvent: "
  //          << "  x(),y() = "  << e->x() << ", " << e->y()
  //          << '\n';
  //m_point2->setX(e->x());
  //m_point2->setY(e->y());
  update();
}

//--------------------------
//--------------------------
//----   Test images   -----
//--------------------------
//--------------------------

void 
WdgImageFigs::onTest()
{
  MsgInLog(_name_(), INFO, "onTest() - slot");
  //std::cout << "WdgImageFigs::onTest() - slot\n";  
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------

} // namespace PSQt

//--------------------------
