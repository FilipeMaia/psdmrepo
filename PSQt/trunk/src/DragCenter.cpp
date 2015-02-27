//--------------------------

#include "PSQt/DragCenter.h"

namespace PSQt {

//--------------------------

DragCenter::DragCenter(WdgImage* wimg, const QPointF* points)
  : DragBase(wimg, points, 1)
{
  //std::cout << "c-tor DragCenter()  xc:" << points[0].x() << "  yc:" << points[0].y() << '\n';
  //std::cout << "DragCenter() c-tor\n";
}

//--------------------------
void
DragCenter::draw(const DRAGMODE& mode)
{
  //std::cout << "DragCenter::draw()\n";

  setImagePointsFromRaw();

  float rad = 20;
  float dev = 5;

  QPointF pc = m_points_img[0];

  //std::cout << "DragCenter::draw()  xc:" << pc.x() << "  yc:" << pc.y() << '\n';

  QPointF points[6] = { 
    pc,
    pc + QPointF(0,-rad),
    pc + QPointF(0,rad),
    pc + QPointF(-dev,dev),
    pc + QPointF(-rad,0),
    pc + QPointF(rad,0)
 	              };

  QPainter* painter = m_wimg->getPainter();
  if (mode == MOVE) painter->setPen(m_pen_move);
  else painter->setPen(m_pen_draw);
  painter->drawPolyline(&points[0], 6);
}

//--------------------------
void
DragCenter::create()
{
  std::cout << "DragCenter::create()\n";
}

//--------------------------
bool
DragCenter::contains(const QPointF& p)
{
  QPointF d = p - m_points_img[0];

  if (d.manhattanLength() < m_rpick) return true;
  return false;
}

//--------------------------
void
DragCenter::move(const QPointF& p)
{
  //std::cout << "DragCenter::move()\n";

  m_points_img[0].setX(p.x());
  m_points_img[0].setY(p.y());

  setRawPointsFromImage();
}

//--------------------------
void
DragCenter::move_is_completed(const QPointF& p)
{
  this->move(p); // last move

  stringstream ss; ss << "Center x: " << fixed << std::setprecision(1) << m_points_raw[0].x()
                                                            << "  y: " << m_points_raw[0].y();
  MsgInLog(_name_(), INFO, ss.str() );  
}

//--------------------------
//const QPointF& 
//DragCenter::getCenter()
//{ 
//  return m_points_raw[0]; 
//}

//--------------------------
void
DragCenter::print()
{
  stringstream ss; ss << strTimeStamp() << " DragCenter::print():";
  std::cout << ss.str() << '\n';
}

//--------------------------

} // namespace PSQt

//--------------------------


