//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------

//--------------------------

#include "PSQt/DragCircle.h"
#include "PSQt/WdgImageFigs.h"

namespace PSQt {

//--------------------------

DragCircle::DragCircle(WdgImage* wimg, const QPointF* points)
  : DragBase(wimg, points, 2)
{
  //std::cout << "DragCircle() c-tor\n";
  m_rad_raw = points[1] - points[0];
}

//--------------------------

const QPointF& 
DragCircle::getCenter()
{ 
  return ((WdgImageFigs*)m_wimg)->getCenter();
}

//--------------------------
void
DragCircle::draw(const DRAGMODE& mode)
{
  //std::cout << "DragCircle::draw()\n";

  m_points_raw[0] = this->getCenter();
  m_points_raw[1] = m_points_raw[0] + m_rad_raw;

  setImagePointsFromRaw();

  QPointF pc = m_points_img[0];
  QPointF dr = m_points_img[1] - m_points_img[0];

  //QPointF dp = pr-pc;
  //float  rad = sqrt(dp.x()*dp.x() + dp.y()*dp.y());

  QPainter* painter = m_wimg->getPainter();
  if (mode == MOVE) painter->setPen(m_pen_move);
  else painter->setPen(m_pen_draw);
  
  painter->drawEllipse(pc, dr.x(), dr.y());
}

//--------------------------
void
DragCircle::create()
{
  std::cout << "DragCircle::create()\n";
}

//--------------------------
bool
DragCircle::contains(const QPointF& p)
{
  QPointF dp = m_wimg->pointInRaw(p) - m_points_raw[0];
  float   r_click = sqrt(dp.x()*dp.x() + dp.y()*dp.y());

  if ( fabs(r_click-m_rad_raw.x()) < m_rpick ) {
    //std::cout << "DragCircle::contains()\n";
    return true;
  }
  return false;
}

//--------------------------
void
DragCircle::move(const QPointF& p)
{
  //std::cout << "DragCircle::move()\n";
  QPointF dp = m_wimg->pointInRaw(p) - m_points_raw[0];
  float r_click = sqrt(dp.x()*dp.x() + dp.y()*dp.y());
  m_rad_raw = QPointF(r_click,r_click);
}

//--------------------------
void
DragCircle::moveIsCompleted(const QPointF& p)
{
  this->move(p); // last move

  stringstream ss; ss << "Radius: " << fixed << std::setprecision(1) << m_rad_raw.x();
  MsgInLog(_name_(), INFO, ss.str() );  
}

//--------------------------
void
DragCircle::print()
{
  stringstream ss; ss << strTimeStamp() << " DragCircle::print():";
  std::cout << ss.str() << '\n';
}

//--------------------------

} // namespace PSQt

//--------------------------


