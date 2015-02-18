#ifndef FRAME_H
#define FRAME_H

//#include <QVBox>
//#include <QWidget>
//#include <QFrame>
//#include <QLabel>
//#include <QSlider>
//#include <QPushButton>

#include <Qt>
#include <QtGui>
#include <QtCore>

#include "PSQt/WdgImage.h"
#include "PSQt/WdgFile.h"

namespace PSQt {

class Frame : public QFrame
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
    Frame( QWidget *parent = 0, Qt::WindowFlags f = 0 );

    void setFrame() ;
    void showTips() ;

    //void resizeEvent     (QResizeEvent *event = 0) ;
    //void closeEvent      (QCloseEvent  *event = 0) ;
    //void moveEvent       (QMoveEvent   *event = 0) ;
    //void mousePressEvent (QMouseEvent  *event = 0) ;

 private:

};

} // namespace PSQt

#endif // FRAME_H
