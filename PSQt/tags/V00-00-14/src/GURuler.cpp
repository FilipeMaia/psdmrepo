//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------

//--------------------------
#include "PSQt/GURuler.h"
#include "PSQt/QGUtils.h"

#include <QFont>
#include <QGraphicsTextItem>

//#include <iostream>    // for std::cout
//#include <fstream>     // for std::ifstream(fname)
//#include <math.h>  // atan2
//#include <cstring> // for memcpy

//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------
GURuler::GURuler( QGraphicsView& view
		, const GURuler::ORIENT& orient
                , const float& vmin
                , const float& vmax
                , const float& vort
                , const unsigned& ndiv1
                , const unsigned& ndiv2 
	        , const unsigned& opt
                , const int& txt_off_h
                , const int& txt_off_v
                , const int& size_tick1
                , const int& size_tick2
	        , const QColor& color
	        , const QPen& pen
	        , const QFont& font
) : QWidget()
  , m_view(view)
  , m_orient(orient)
  , m_vmin(vmin)
  , m_vmax(vmax)
  , m_vort(vort)
  , m_opt(opt)
  , m_txt_off_h(txt_off_h)
  , m_txt_off_v(txt_off_v)
  , m_color(color)
  , m_pen(pen)
  , m_font(font)
  , m_path()
{
  m_pen.setCosmetic(true);
  m_pen.setColor(m_color);

  m_ndiv1 = (ndiv1) ? ndiv1 : 1;
  m_ndiv2 = (ndiv2) ? ndiv2 : 1;

  m_size_tick1 = (size_tick1) ? size_tick1 : 8;
  m_size_tick2 = (size_tick2) ? size_tick2 : 4;

  v_textitems.clear();

  setPars();
  setPathForRuler();
}

//--------------------------

GURuler::~GURuler() 
{
      m_view.scene()->removeItem((QGraphicsItem*) m_path_item);

      for(std::vector<QGraphicsItem*>::iterator it=v_textitems.begin(); it!=v_textitems.end(); ++it) 
         m_view.scene()->removeItem(*it);

      delete m_path_item;

      v_textitems.clear();
}


//--------------------------
//--------------------------

void 
GURuler::setPars()
  {
    m_scx = m_view.transform().m11();
    m_scy = m_view.transform().m22();

    switch (m_orient) {

      default : 
      case HU : 
        m_p1   = QPointF(m_vmin, m_vort);
        m_p2   = QPointF(m_vmax, m_vort);
    	m_dt1  = QPointF(0, m_size_tick1/m_scy);
    	m_dt2  = QPointF(0, m_size_tick2/m_scy);
        m_hoff = -4/m_scx;
        m_dtxt = QPointF(-4/m_scx,-25/m_scy);
	return;

      case HD : 
        m_p1   = QPointF(m_vmin, m_vort);
        m_p2   = QPointF(m_vmax, m_vort);
    	m_dt1  = QPointF(0, -m_size_tick1/m_scy);
    	m_dt2  = QPointF(0, -m_size_tick2/m_scy);
        m_hoff = -4/m_scx;
        m_dtxt = QPointF(-4/m_scx,0/m_scy);
	return;
 
      case VL : 
        m_p1   = QPointF(m_vort, m_vmin);
        m_p2   = QPointF(m_vort, m_vmax);
    	m_dt1  = QPointF(m_size_tick1/m_scx, 0);
    	m_dt2  = QPointF(m_size_tick2/m_scx, 0);
        m_hoff = -8/m_scx;
        m_dtxt = QPointF(-20/m_scx,-15/m_scy);

	return;

      case VR : 
        m_p1   = QPointF(m_vort, m_vmin);
        m_p2   = QPointF(m_vort, m_vmax);
    	m_dt1  = QPointF(-m_size_tick1/m_scx, 0);
    	m_dt2  = QPointF(-m_size_tick2/m_scx, 0);
        m_hoff = 0/m_scx;
        m_dtxt = QPointF(0,-15/m_scy);
	return;
  }
}

//--------------------------

  void GURuler::printTransform()
  {
    QTransform trans = m_view.transform();
    std::cout << "GURuler::printTransform():"
              << "\n m11():" << trans.m11()
    	      << "\n m22():" << trans.m22()
    	      << "\n dx() :" << trans.dx()
    	      << "\n dy() :" << trans.dy()
              << '\n';
  }

//--------------------------

void 
GURuler::setPathForRuler()
  {
    //QFont font(m_font_name.c_str(), m_font_size, false);
    //font.setRawMode(true);

    // Add ruller scale
    m_path.moveTo(m_p2);
    m_path.lineTo(m_p1);   

    QPointF dp((m_p2-m_p1)/m_ndiv1);
    QPointF dpf(dp/m_ndiv2);
    QPointF pc(m_p1);
    QPointF pc_pix;

    for(unsigned i=0; i<=m_ndiv1; i++) {
      m_path.moveTo(pc);
      m_path.lineTo(pc+m_dt1);
      
      if(i==m_ndiv1) break;

      for(unsigned j=0; j<m_ndiv2; j++) {      
        pc += dpf;
        m_path.moveTo(pc);
        m_path.lineTo(pc+m_dt2);
      }
    }

    m_path_item = m_view.scene()->addPath(m_path, m_pen);

    // Add labels

    //this->printTransform();

    if(! m_opt & 1) return;

    v_textitems.clear();

    pc=m_p1;
    for(unsigned i=0; i<=m_ndiv1; i++, pc += dp) {
      float val = (m_orient==HD || m_orient==HU) ? pc.x() : pc.y();

      QGraphicsTextItem* txtitem = m_view.scene()->addText(QString(val_to_string<float>(val,0).c_str()), m_font);
      QString qstr = txtitem->toPlainText();
      //std::cout << "QString: " << qstr.toStdString() << "  size: " << qstr.size() << '\n';

      txtitem->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);

      //QPoint pv(m_view.mapFromScene(pc));
      //txtitem->setPos(m_view.mapToScene(pv + m_dtxt + QPoint(m_hoff*qstr.size(),0)));

      txtitem->setPos(pc + m_dtxt + QPointF(m_hoff*qstr.size(),0));

      txtitem->setDefaultTextColor(m_color);
      txtitem->adjustSize();

      v_textitems.push_back((QGraphicsItem*)txtitem);
    }
  }

//--------------------------

//--------------------------
} // namespace PSQt
//--------------------------
