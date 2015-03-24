#ifndef WDGCOLORBAR_H
#define WDGCOLORBAR_H

#include <QWidget>
#include <QLabel>
#include <QPainter>
#include <QPixmap>

#include <Qt>
#include <QtGui>
#include <QtCore>

#include "PSQt/QGUtils.h"
#include "ndarray/ndarray.h"

namespace PSQt {

/**
 *  @ingroup PSQt
 * 
 *  @brief WdgColorBar - QLable widget showing interactive color bar 
 * 
 *  Parameters of the color bar are set through the constructor, public method setColorBar, 
 *  and slot: onHueAnglesUpdated(const float&, const float&)
 *
 *  Emits signal on mouse button press/release/move:
 *     void pressColorBar(QMouseEvent*, const float&);
 *     void releaseColorBar(QMouseEvent*, const float&);
 *     void moveOnColorBar(QMouseEvent*, const float&);
 * 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see WdgColorTable.h
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

enum ORIENT {HR=0, HL, VU, VD};

class WdgColorBar : public QLabel
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
  /**
   *  @brief Class constructor with parameters
   *  
   *  @param[in] h1 - 1st boarder hue angle.
   *  @param[in] h2 - 2nd boarder hue angle.
   *  @param[in] colors - number of colors in the color bar.
   *  @param[in] orient - orientation from the numerated list: HR, HL, VU, VD
   *  @param[in] aspect - alpect ratio; width/length
   */ 
  WdgColorBar( QWidget *parent=0, 
               const float& h1=260, 
               const float& h2=-60, 
               const unsigned& colors=1024, 
               const ORIENT& orient=HR, 
               const float& aspect=0.03);

  virtual ~WdgColorBar();

    void setStyle() ;
    void showTips() ;

    void closeEvent (QCloseEvent  *event = 0) ;
    void moveEvent  (QMoveEvent   *event = 0) ;

    void setColorBar( const float&    h1=0,
                      const float&    h2=360,
                      const unsigned& colors=1024, 
                      const ORIENT&   orient=HR, 
                      const float&    aspect=0.03
                    );

 protected :
    void mousePressEvent   (QMouseEvent  *event = 0) ;
    void mouseReleaseEvent (QMouseEvent  *event = 0) ;
    void mouseMoveEvent    (QMouseEvent  *event = 0) ;
    void resizeEvent       (QResizeEvent *event = 0) ;

 signals :
    void pressColorBar(QMouseEvent*, const float&) ;
    void releaseColorBar(QMouseEvent*, const float&) ;
    void moveOnColorBar(QMouseEvent*, const float&) ;

 public slots:
    void onHueAnglesUpdated(const float&, const float&) ;
    void testPressColorBar(QMouseEvent*, const float&) ;
    void testReleaseColorBar(QMouseEvent*, const float&) ;
    void testMoveOnColorBar(QMouseEvent*, const float&) ;

 private:
    float         m_h1;
    float         m_h2;
    unsigned      m_colors;
    ORIENT        m_orient;
    float         m_aspect; 

    QLabel*       m_lab_cbar;
    QPixmap*      m_pixmap_cbar;

    inline const char* _name_(){return "WdgColorBar";}
    float fractionOfColorBar(QMouseEvent *e);
    void connectForTest(); 
    void message(QMouseEvent* e, const float& frac, const char* comment);
};

} // namespace PSQt

#endif // WDGCOLORBAR_H
