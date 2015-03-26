#ifndef WDGCOLORTABLE_H
#define WDGCOLORTABLE_H

#include <QWidget>
#include <QLabel>
#include <QPainter>
#include <QPixmap>

#include <Qt>
#include <QtGui>
#include <QtCore>

#include "PSQt/LabColorRing.h"
#include "PSQt/QGUtils.h"
#include "ndarray/ndarray.h"

namespace PSQt {

/**
 *  @ingroup PSQt
 * 
 *  @brief Widget showing colot wheel and setting parameters (hue angles) for color table.
 * 
 *  @code
 *  @endcode
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see GUIImageViewer
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

class WdgColorTable : public QWidget
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
    WdgColorTable( QWidget *parent = 0, const float& h1=260, const float& h2=-60, const unsigned& colors=1024 );

    void setFrame() ;
    void setStyle() ;
    void showTips() ;

    void closeEvent      (QCloseEvent  *event = 0) ;
    void moveEvent       (QMoveEvent   *event = 0) ;

    void setColorBar( const float&    hue1=0,
                      const float&    hue2=360,
                      const unsigned& rows=40,
                      const unsigned& cols=256
                      );

    float getH1() {return m_h1;}
    float getH2() {return m_h2;}

    uint32_t* getColorTable(const unsigned& colors=1024) { return ColorTable(colors, m_h1, m_h2); }
    uint32_t* getColorTable(const unsigned& colors, const float& h1, const float& h2) { 
              return ColorTable(colors, h1, h2); }

    ndarray<uint32_t,1> getColorTableAsNDArray(const unsigned& colors=1024);

 protected :

    void mousePressEvent (QMouseEvent  *event = 0) ;
    void mouseMoveEvent  (QMouseEvent  *event = 0) ;
    void resizeEvent     (QResizeEvent *event = 0) ;

    //bool eventFilter     (QObject *obj, QEvent *event = 0) ;


 signals :

    void hueAnglesUpdated(const float&, const float&) ;
    void hueAngleIsEdited(const unsigned& edited) ;

 public slots:

    void onButExit() ;
    void onEdiH1() ;
    void onEdiH2() ;
    void onButApply() ;
    void onSetH(const unsigned& selected) ;
    void onHueAngleIsChanged() ;
    void testSignalHueAnglesUpdated(const float&, const float&) ;

 private:

    float         m_h1;
    float         m_h2;
    unsigned      m_colors;
    unsigned      m_figsize;
    unsigned      m_cbar_width;

    QFrame*       m_frame;
    LabColorRing* m_lab_cring;
    QLabel*       m_lab_cbar;
    QLabel*       m_lab_h1;
    QLabel*       m_lab_h2;
    QLineEdit*    m_edi_h1;
    QLineEdit*    m_edi_h2;
    QPushButton*  m_but_apply;

    QPixmap*      m_pixmap_cbar;
    inline const char* _name_(){return "WdgColorTable";}
};

} // namespace PSQt

#endif // WDGCOLORTABLE_H
