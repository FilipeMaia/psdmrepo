#ifndef GUIIMAGEVIEWER_H
#define GUIIMAGEVIEWER_H

#include "PSQt/Frame.h"
//#include <QVBox>
//#include <QWidget>
//#include <QLabel>
//#include <QSlider>
//#include <QPushButton>

#include <Qt>
#include <QtGui>
#include <QtCore>

//#include "PSQt/WdgImage.h"
#include "PSQt/WdgImageFigs.h"
#include "PSQt/WdgFile.h"

namespace PSQt {
class GUIImageViewer : public Frame
//class GUIImageViewer : public QWidget
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
    GUIImageViewer( QWidget *parent = 0 );

    void resizeEvent     (QResizeEvent *event = 0) ;
    void closeEvent      (QCloseEvent  *event = 0) ;
    void moveEvent       (QMoveEvent   *event = 0) ;
    void mousePressEvent (QMouseEvent  *event = 0) ;
    //inline WdgImageFigs* wdgImageFigs(){ return m_image; }
    inline WdgImage* wdgImage(){ return (WdgImage*)m_image; }
    inline WdgFile*  wdgFile() { return m_file; }

 public slots:

    void onButExit() ;
    void onButAdd() ;

 private:

    //QLabel*      m_lab_fname;
    //QLineEdit*   m_edi_fncfg;
    QPushButton*   m_but_exit;
    QPushButton*   m_but_add;
    //WdgImage*      m_image;
    WdgImageFigs*  m_image;
    WdgFile*       m_file;

    inline const char* _name_(){ return "GUIImageViewer"; }
    void showTips() ;
};

} // namespace PSQt

#endif // GUIIMAGEVIEWER_H
