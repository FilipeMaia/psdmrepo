//--------------------------

#include "PSQt/DragBase.h"

using namespace std;   // for cout without std::

namespace PSQt {

//--------------------------

DragBase::DragBase(WdgImage* wimg, const QPointF* points, const int& npoints)
  : m_wimg(wimg)
  , m_npoints(npoints)
  , m_pen_draw(Qt::white, 1, Qt::SolidLine)
  , m_pen_move(Qt::red,   2, Qt::SolidLine) // Qt::DashLine)
  , m_rpick(10)
  , m_center_def(QPointF(100,100)) 
{
  //std::cout << "c-tor DragBase()\n";
  m_points_raw = new QPointF[m_npoints];
  m_points_img = new QPointF[m_npoints];

  this->copyRawPoints(points);
  this->setImagePointsFromRaw();
}

//--------------------------
DragBase::~DragBase()
{
  delete [] m_points_raw;
  delete [] m_points_img;
}; 

//--------------------------

void
DragBase::copyRawPoints(const QPointF* points)
{
  for(int i=0; i<m_npoints; ++i) m_points_raw[i] = points[i];
}

//--------------------------

void
DragBase::setImagePointsFromRaw()
{
  for(int i=0; i<m_npoints; ++i) m_points_img[i] = m_wimg->pointInImage(m_points_raw[i]);
}

//--------------------------

void
DragBase::setRawPointsFromImage()
{
  for(int i=0; i<m_npoints; ++i) m_points_raw[i] = m_wimg->pointInRaw(m_points_img[i]);
}

//--------------------------

void
DragBase::print()
{
  stringstream ss; ss << strTimeStamp() << "  DragBase::print():";
  for(int i=0; i<m_npoints; ++i) ss <<"\n    x_raw:" << m_points_raw[i].x() << "  y_raw:" << m_points_raw[i].y();
  std::cout << ss.str() << '\n';
}

//--------------------------

} // namespace PSQt

//--------------------------


