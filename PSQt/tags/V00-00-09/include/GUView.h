#ifndef GUVIEW_H
#define GUVIEW_H

//#include <string>

#include <QGraphicsView>
//#include <QWidget>
//#include <QTransform>
//#include <QGraphicsScene>
//#include <QPoint>
//#include <QPainterPath>
//#include <QPen>
//#include <QBrash>

namespace PSQt {

//--------------------------

class GUView // : public QWidget
{
  //Q_OBJECT // macro is needed for connection of signals and slots

public:

  //GUView () {}
  //virtual ~GUView () {}
  //enum ORIDIR {HORUP, HORDOWN, VERRIGHT, VERLEFT};

  static QGraphicsView*
  make_view( const float& xmin =   0
           , const float& xmax =  30
           , const float& ymin = -10
           , const float& ymax =  20
           , const unsigned pribits = 0177777 ); 

  //private:

    //std::string m_oristr;
    //float       m_xmin;
    //float       m_xmax;
    //float       m_ymin;
    //float       m_ymax;
};

//--------------------------
} // namespace PSQt

#endif // GUVIEW_H
