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
#include "PSQt/WdgPointPos.h"
#include "PSQt/WdgColorTable.h"
#include "PSQt/ImageProc.h"
#include "PSQt/WdgRadHist.h"
#include "PSQt/WdgSpecHist.h"

namespace PSQt {

//--------------------------

/**
 *  @ingroup PSQt
 * 
 *  @brief Widget/GUI for image viewer.
 * 
 *  @code
 *  @endcode
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see GUIMain
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

//--------------------------

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
    inline ImageProc* getImageProc() { return m_imageproc; }
    void showChildWindows();

 public slots:
    void onButExit(); // is not used
    void onButSpec();
    void onButRHis();
    void onButAdd();
    void onButColorTab();
    void onImageIsUpdated(ndarray<GeoImage::raw_image_t,2>&);

 private:
    //QLabel*      m_lab_fname;
    //QLineEdit*   m_edi_fncfg;
    QPushButton*   m_but_spec;
    QPushButton*   m_but_rhis;
    QPushButton*   m_but_add;
    QPushButton*   m_but_cols;
    QHBoxLayout*   m_hbox;
    QVBoxLayout*   m_vbox;

    //WdgImage*      m_image;
    WdgImageFigs*    m_image;
    WdgFile*         m_file;
    WdgPointPos*     m_pointpos;
    WdgColorTable*   m_colortab;
    ImageProc*       m_imageproc;
    WdgRadHist*      m_radhist;
    WdgSpecHist*     m_spechist;

    inline const char* _name_(){ return "GUIImageViewer"; }
    void setStyle();
    void setTips();
    void message(QWidget* wdg, const char* cmt);
};

} // namespace PSQt

#endif // GUIIMAGEVIEWER_H
