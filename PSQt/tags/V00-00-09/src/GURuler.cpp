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
		, const std::string& oristr
                , const float& vmin
                , const float& vmax
                , const float& vort
                , const unsigned& ndiv1
                , const unsigned& ndiv2 
                , const int& txt_off_h
                , const int& txt_off_v
	        , const unsigned& opt
) : QWidget()
  , m_view(view)
  , m_oristr(oristr)
  , m_vmin(vmin)
  , m_vmax(vmax)
  , m_vort(vort)
  , m_txt_off_h(txt_off_h)
  , m_txt_off_v(txt_off_v)
  , m_opt(opt)
  , m_path()
{
  if     (oristr == std::string("HU")) m_orient = HORUP;
  else if(oristr == std::string("HD")) m_orient = HORDOWN;
  else if(oristr == std::string("VL")) m_orient = VERLEFT;
  else if(oristr == std::string("VR")) m_orient = VERRIGHT;
  else                                 m_orient = HORUP;

  m_ndiv1 = (ndiv1) ? ndiv1 : 1;
  m_ndiv2 = (ndiv2) ? ndiv2 : 1;
  m_size_tick1 =  8;
  m_size_tick2 =  4;
  m_font_size  = 12;
  m_font_name  = "Helvetica [Cronyx]"; // "Monospace", "Tepewritter" "Sans Serif"
  m_scx = m_view.transform().m11();
  m_scy = m_view.transform().m22();

  setPars();
  setPathForRuler();
}

//--------------------------
//--------------------------
//--------------------------

void 
GURuler::setPars()
  {
    switch (m_orient) {

      default : 
      case HORUP : 
        m_p1   = QPointF(m_vmin, m_vort);
        m_p2   = QPointF(m_vmax, m_vort);
    	m_dt1  = QPointF(0, m_size_tick1/m_scy);
    	m_dt2  = QPointF(0, m_size_tick2/m_scy);
    	m_dtxt = QPointF((-5+m_txt_off_h)/m_scx, (-20-m_txt_off_v)/m_scy);
	return;

      case HORDOWN : 
        m_p1   = QPointF(m_vmin, m_vort);
        m_p2   = QPointF(m_vmax, m_vort);
    	m_dt1  = QPointF(0, -m_size_tick1/m_scy);
    	m_dt2  = QPointF(0, -m_size_tick2/m_scy);
    	m_dtxt = QPointF((-5+m_txt_off_h)/m_scx, (-4-m_txt_off_v)/m_scy);
	return;
 
      case VERLEFT : 
        m_p1   = QPointF(m_vort, m_vmin);
        m_p2   = QPointF(m_vort, m_vmax);
    	m_dt1  = QPointF(m_size_tick1/m_scx, 0);
    	m_dt2  = QPointF(m_size_tick2/m_scx, 0);
    	m_dtxt = QPointF((-22+m_txt_off_h)/m_scx, (-15-m_txt_off_v)/m_scy);
	return;

      case VERRIGHT : 
        m_p1   = QPointF(m_vort, m_vmin);
        m_p2   = QPointF(m_vort, m_vmax);
    	m_dt1  = QPointF(-m_size_tick1/m_scx, 0);
    	m_dt2  = QPointF(-m_size_tick2/m_scx, 0);
	m_dtxt = QPointF((2+m_txt_off_h)/m_scx, (-15-m_txt_off_v)/m_scy);
	return;
  }
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

    // Add labels

    if(! m_opt & 1) return;

    pc=m_p1;
    for(unsigned i=0; i<=m_ndiv1; i++, pc += dp) {
      float val = (m_orient==HORDOWN || m_orient==HORUP) ? pc.x() : pc.y();
      //m_path.addText(pc+m_dtxt, font, QString(val_to_string<float>(val).c_str()) ); 
      QGraphicsTextItem* txtitem = m_view.scene()->addText(QString(val_to_string<float>(val).c_str()), QFont());
      txtitem->setFlag(QGraphicsItem::ItemIgnoresTransformations, true);
      txtitem->setPos(pc+m_dtxt);
    }
  }

//--------------------------

//--------------------------
} // namespace PSQt
//--------------------------
