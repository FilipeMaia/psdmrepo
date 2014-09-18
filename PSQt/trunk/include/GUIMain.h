#ifndef GUIMAIN_H
#define GUIMAIN_H

//#include <QVBox>
//#include <QWidget>
//#include <QLabel>
//#include <QSlider>
//#include <QPushButton>

#include <Qt>
#include <QtGui>
#include <QtCore>

#include "PSQt/WdgImage.h"
#include "PSQt/WdgFile.h"

namespace PSQt {

class GUIMain : public QWidget
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
    GUIMain( QWidget *parent = 0 );

    void setFrame() ;
    void showTips() ;

    void resizeEvent     (QResizeEvent *event = 0) ;
    void closeEvent      (QCloseEvent  *event = 0) ;
    void moveEvent       (QMoveEvent   *event = 0) ;
    void mousePressEvent (QMouseEvent  *event = 0) ;

 public slots:

    void onButExit() ;

 private:

    QFrame*      m_frame;

    //QLabel*      m_lab_fname;
    //QLineEdit*   m_edi_fncfg;
    QPushButton* m_but_exit;
    QPushButton* m_but_test;
    WdgImage*    m_image;
    WdgFile*     m_file;
};

} // namespace PSQt

#endif // GUIMAIN_H
