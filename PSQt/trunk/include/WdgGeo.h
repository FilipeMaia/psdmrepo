#ifndef WDGGEO_H
#define WDGGEO_H

#include "PSCalib/GeometryObject.h"
#include <boost/shared_ptr.hpp>

#include <Qt>
#include <QtGui>
#include <QtCore>

#include "PSQt/Frame.h"

namespace PSQt {

/**
 *  @ingroup PSQt
 *
 *  @brief QWidget/Frame for GeometryObject editor.
 * 
 *  WdgGeo is a sub-class of Frame (QWidget).
 *
 *  WdgGeo is an interactive editor of the GeometryObject parameters.
 *  Slot setNewGO(shpGO&) - resets the GeometryObject.
 *  Sends signal geoIsChanged() when geo is changed.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see GUIMain
 *
 *  @version $Id:$
 *
 *  @author Mikhail Dubrovin
 */


//class WdgGeo : public QWidget
class WdgGeo : public Frame
{
 Q_OBJECT // macro is needed for connection of signals and slots

 typedef boost::shared_ptr<PSCalib::GeometryObject> shpGO;

 public:
    WdgGeo(QWidget *parent=0, shpGO=shpGO(new PSCalib::GeometryObject()), const unsigned& pbits=0x177777);
    //    ~WdgGeo(){}
    void resizeEvent(QResizeEvent *event = 0) ;
    void moveEvent  (QMoveEvent   *event = 0) ;
    void closeEvent (QCloseEvent  *event = 0) ;

 public slots:
    void setNewGO(shpGO&);
    void onRadioGroup();
    void onButAddSub(QAbstractButton* but);
    void testSignalGeoIsChanged(shpGO&);

 signals :
    void geoIsChanged(shpGO&);

 private :
    std::string   m_path;
    shpGO         m_geo; 
    unsigned      m_pbits;

    QLabel*       m_lab_geo;
    QLabel*       m_lab_par;
    QLineEdit*    m_edi_step;
    QPushButton*  m_but_add;
    QPushButton*  m_but_sub;

    QLineEdit*    m_edi_x0;
    QLineEdit*    m_edi_y0;
    QLineEdit*    m_edi_z0;
    QLineEdit*    m_edi_rot_x;
    QLineEdit*    m_edi_rot_y;
    QLineEdit*    m_edi_rot_z;
    QLineEdit*    m_edi_tilt_x;
    QLineEdit*    m_edi_tilt_y;
    QLineEdit*    m_edi_tilt_z;

    QRadioButton* m_rad_x0;    
    QRadioButton* m_rad_y0;   
    QRadioButton* m_rad_z0;                                  
    QRadioButton* m_rad_rot_x; 
    QRadioButton* m_rad_rot_y; 
    QRadioButton* m_rad_rot_z;                               
    QRadioButton* m_rad_tilt_x;
    QRadioButton* m_rad_tilt_y;
    QRadioButton* m_rad_tilt_z;

    QButtonGroup* m_but_gr;
    QButtonGroup* m_rad_gr;

    QGridLayout*  m_grid;
    QVBoxLayout*  m_box;
    QHBoxLayout*  m_cbox;

    std::map<QRadioButton*,QLineEdit*> map_radio_to_edit;

    inline const std::string _name_(){return "WdgGeo";}
    void setStyle() ;
    void showTips() ;
    void setGeoPars();
    void updateGeoPars();
};

} // namespace PSQt

#endif // WDGGEO_H
