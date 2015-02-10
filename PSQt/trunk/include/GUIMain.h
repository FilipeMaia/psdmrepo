#ifndef GUIMAIN_H
#define GUIMAIN_H

#include "PSCalib/GeometryAccess.h"
#include "PSQt/Frame.h"
#include "PSQt/WdgGeoTree.h"
#include "PSQt/WdgGeo.h"
#include "PSQt/WdgImage.h"
#include "PSQt/WdgFile.h"
#include "PSQt/GeoImage.h"
#include "PSQt/GUILogger.h"

//#include <QVBox>
//#include <QWidget>
//#include <QLabel>
//#include <QSlider>
//#include <QPushButton>

#include <Qt>
#include <QtGui>
#include <QtCore>

namespace PSQt {

/**
 *  @defgroup PSQt PSQt package
 *  @brief Package PSQt is created for graphical Qt applications
 */

/// @addtogroup PSQt PSQt

/**
 *  @ingroup PSQt
 * 
 *  @brief Main GUI of application for detector sensors alignment
 * 
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see WdgFile, WdgGeoTree, WdgGeo
 *
 *  @version $Id:$
 *
 *  @author Mikhail Dubrovin
 */

//class GUIMain : public QWidget
class GUIMain : public Frame
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:

    GUIMain( QWidget *parent = 0 );

    void resizeEvent     (QResizeEvent *event = 0) ;
    void closeEvent      (QCloseEvent  *event = 0) ;
    void moveEvent       (QMoveEvent   *event = 0) ;
    void mousePressEvent (QMouseEvent  *event = 0) ;

    PSCalib::GeometryAccess* geoacc(){return m_wgt->geoacc();}


 public slots:

    void onButExit() ;
    void onButSave() ;

 private:

    WdgGeoTree*  m_wgt;
    WdgGeo*      m_wge;
    QWidget*     m_wrp;
    QWidget*     m_wmbox;

    WdgFile*     m_file_geo;
    WdgFile*     m_file_nda;

    WdgImage*    m_wimage;
    GeoImage*    m_geoimg;

    GUILogger*   m_guilogger;

    QHBoxLayout* m_bbox;
    QVBoxLayout* m_fbox;
    QVBoxLayout* m_rbox;
    QVBoxLayout* m_mbox;
    QVBoxLayout* m_vbox;

    QSplitter*   m_hsplit;
    QSplitter*   m_vsplit;

    QPushButton* m_but_exit;
    QPushButton* m_but_save;

    const std::string _name_(){return "GUIMain";}
    void showTips() ;
    void setStyle() ;
};

} // namespace PSQt

#endif // GUIMAIN_H
