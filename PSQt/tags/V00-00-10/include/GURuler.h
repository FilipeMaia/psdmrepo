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



namespace PSQt {


//--------------------------

class GURuler : public QWidget
{
  Q_OBJECT // macro is needed for connection of signals and slots

public:

  enum ORIDIR {HORUP, HORDOWN, VERRIGHT, VERLEFT};

  GURuler( QGraphicsView& view
         , const std::string& oristr = "HU"
         , const float& vmin    = 0
         , const float& vmax    = 100
         , const float& vort    = 0
         , const unsigned& ndiv1= 5
         , const unsigned& ndiv2= 2
	 , const int& txt_off_h = 0
	 , const int& txt_off_v = 0
	 , const unsigned& opt  = 1
         ); 

  virtual ~GURuler () {}

  void setPars();
  void setPathForRuler();
  //void setPathForRulerV0();

  const QPainterPath& pathForRuler(){ return m_path; }


private:

    QGraphicsView&  m_view;
    std::string     m_oristr;
    float           m_vmin;
    float           m_vmax;
    float           m_vort;
    unsigned        m_ndiv1;
    unsigned        m_ndiv2;
    float           m_scx;
    float           m_scy;

    ORIDIR          m_orient;

    QPointF         m_p1;
    QPointF         m_p2;
    QPointF         m_dt1;
    QPointF         m_dt2;
    QPointF         m_dtxt;

    int m_size_tick1;
    int m_size_tick2;
    int m_txt_off_h;
    int m_txt_off_v;
    unsigned m_opt;
    unsigned m_font_size;
    std::string m_font_name;

    QPainterPath m_path;
};

//--------------------------
} // namespace PSQt

#endif // GURULER_H
