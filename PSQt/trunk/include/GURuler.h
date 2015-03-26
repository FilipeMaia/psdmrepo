#ifndef GURULER_H
#define GURULER_H

#include <string>

//#include <QPen>
//#include <QBrash>
#include <QWidget>
#include <QPoint>
#include <QPainterPath>
#include <QGraphicsView>
//#include <QGraphicsScene>
//#include <QTransform>
//#include <QTextDocument>


namespace PSQt {

//--------------------------

/**
 *  @ingroup PSQt
 * 
 *  @brief Shows ruller on QGraphicsScene through QGraphicsView.
 * 
 *  @code
 *  @endcode
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see GUAxes
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

//--------------------------

class GURuler : public QWidget
{
  Q_OBJECT // macro is needed for connection of signals and slots

public:

  enum ORIENT {HD=0, HU, VL, VR};

  GURuler( QGraphicsView& view
         , const ORIENT& orient = HD
         , const float& vmin    = 0
         , const float& vmax    = 100
         , const float& vort    = 0
         , const unsigned& ndiv1= 5
         , const unsigned& ndiv2= 2
	 , const unsigned& opt  = 1
	 , const int& txt_off_h = 0
	 , const int& txt_off_v = 0
         , const int& size_tick1= 8
         , const int& size_tick2= 4
	 , const QColor& color  = QColor(Qt::black)
	 , const QPen& pen      = QPen(Qt::black, 2, Qt::SolidLine)
	 , const QFont& font    = QFont()
         ); 

  virtual ~GURuler();

  void setPars();
  void setPathForRuler();
  //void setPathForRulerV0();

  const QPainterPath& pathForRuler(){ return m_path; }
  void printTransform();


private:

    QGraphicsView&  m_view;
    ORIENT          m_orient;
    float           m_vmin;
    float           m_vmax;
    float           m_vort;
    unsigned        m_ndiv1;
    unsigned        m_ndiv2;
    unsigned        m_opt;
    int             m_txt_off_h;
    int             m_txt_off_v;
    int             m_size_tick1;
    int             m_size_tick2;
    QColor          m_color;
    QPen            m_pen;
    QFont           m_font;

    //QTextOption  m_textop;
    std::vector<QGraphicsItem*> v_textitems;

    float           m_scx;
    float           m_scy;
    float           m_hoff;

    QPainterPath       m_path;
    QGraphicsPathItem* m_path_item;

    QPointF         m_p1;
    QPointF         m_p2;
    QPointF         m_dt1;
    QPointF         m_dt2;
    QPointF         m_dtxt;
};

//--------------------------
} // namespace PSQt

#endif // GURULER_H
