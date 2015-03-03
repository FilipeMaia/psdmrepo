//--------------------------

#include "PSQt/GUIMain.h"
#include "PSQt/Logger.h"
#include "PSQt/QGUtils.h"
#include "AppUtils/AppDataPath.h"

//#include <string>
//#include <fstream>   // ofstream
//#include <iomanip>   // for setw, setfill
//#include <math.h>
//#include <stdio.h>

#include <sstream>   // for stringstream
#include <iostream>    // cout
#include <fstream>    // ifstream(fname), ofstream
//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------


GUIMain::GUIMain(QWidget *parent, const LEVEL& level)
    : Frame(parent)
//  : QWidget(parent)
{
  MsgInLog(_name_(), INFO, "Create the main control window for this app."); 

  const std::string base_dir = "/reg/g/psdm/detector/alignment/cspad/calib-cxi-ds1-2014-05-15/";
  const std::string fname_geo = base_dir + "calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data"; 
  const std::string fname_nda = base_dir + "cspad-arr-cxid2714-r0023-lysozyme-rings.txt"; 
  const bool pbits = 0;

  AppUtils::AppDataPath adp_icon_exit("PSQt/icons/exit.png");
  AppUtils::AppDataPath adp_icon_save("PSQt/icons/save.png");

  m_but_exit = new QPushButton( "Exit", this );
  m_but_save = new QPushButton( "Save", this );

  m_but_exit -> setIcon(QIcon(QString(adp_icon_exit.path().c_str())));
  m_but_save -> setIcon(QIcon(QString(adp_icon_save.path().c_str()))); 

  m_file_geo = new PSQt::WdgFile(this, "Set geometry", fname_geo, "*.data \n *", false);
  m_file_nda = new PSQt::WdgFile(this, "Set ndarray",  fname_nda, "*.txt *.dat \n *", false);

  m_wgt = new PSQt::WdgGeoTree(this, fname_geo, pbits);
  m_wge = new PSQt::WdgGeo(this);
 
  m_bbox = new QHBoxLayout();
  m_bbox -> addWidget(m_but_save);
  m_bbox -> addStretch(1);
  m_bbox -> addWidget(m_but_exit);

  m_fbox = new QVBoxLayout();
  m_fbox -> addWidget(m_file_geo);
  m_fbox -> addWidget(m_file_nda);

  m_rbox = new QVBoxLayout();
  m_rbox -> addWidget(m_wge);
  m_rbox -> addStretch(1);
  m_rbox -> addLayout(m_bbox);

  m_wrp = new QWidget();
  m_wrp -> setLayout(m_rbox);
  m_wrp -> setContentsMargins(-9,-9,-9,-9);

  m_hsplit = new QSplitter(Qt::Horizontal);
  m_hsplit -> addWidget(m_wgt); 
  m_hsplit -> addWidget(m_wrp);

  m_mbox = new QVBoxLayout();
  m_mbox -> addLayout(m_fbox);
  m_mbox -> addWidget(m_hsplit); 

  m_wmbox = new QWidget();
  m_wmbox -> setLayout(m_mbox);
  m_wmbox -> setContentsMargins(-9,-9,-9,-9);

  m_guilogger = new PSQt::GUILogger(this, false); // true/false - show/do not show buttons
  m_guilogger -> setContentsMargins(-9,-9,-9,-9);

  m_vsplit = new QSplitter(Qt::Vertical); 
  m_vsplit -> addWidget(m_wmbox); 
  m_vsplit -> addWidget(m_guilogger); 

  m_vbox = new QVBoxLayout();
  m_vbox -> addWidget(m_vsplit);

  this -> setLayout(m_vbox);

  showTips();
  setStyle();

  m_geoimg = new PSQt::GeoImage(m_wgt->geoacc(), fname_nda);  

  m_guiimv = new PSQt::GUIImageViewer(0);
  m_guiimv -> move(this->pos().x() + this->size().width() + 8, this->pos().y());  
  m_guiimv -> show();
  m_wimage = m_guiimv->wdgImage();

  //  m_wimage = new PSQt::WdgImage(0);  
  //  m_wimage -> move(this->pos().x() + this->size().width() + 8, this->pos().y());  
  //  m_wimage -> show();

  connect(Logger::getLogger(), SIGNAL(signal_new_record(Record&)), m_guilogger, SLOT(addNewRecord(Record&)));

  connect(m_wgt->get_view(), SIGNAL(selectedGO(shpGO&)), m_wge, SLOT(setNewGO(shpGO&)));
  connect(m_but_exit, SIGNAL( clicked() ), this, SLOT(onButExit()));
  connect(m_but_save, SIGNAL( clicked() ), this, SLOT(onButSave()));
  connect(m_file_geo, SIGNAL(fileNameIsChanged(const std::string&)), m_wgt->get_view(), SLOT(updateTreeModel(const std::string&))); 

  // connect signals for image update

  connect(m_wgt->get_view(), SIGNAL(geometryIsLoaded(PSCalib::GeometryAccess*)), m_geoimg, SLOT(onGeometryIsLoaded(PSCalib::GeometryAccess*))); 
  connect(m_file_nda, SIGNAL(fileNameIsChanged(const std::string&)), m_geoimg, SLOT(onImageFileNameIsChanged(const std::string&))); 
  connect(m_wge,      SIGNAL(geoIsChanged(shpGO&)), m_geoimg, SLOT(onGeoIsChanged(shpGO&)));
  //connect(m_geoimg, SIGNAL(normImageIsUpdated(const ndarray<GeoImage::image_t,2>&)), m_wimage, SLOT(onNormImageIsUpdated(const ndarray<GeoImage::image_t,2>&)));
  connect(m_geoimg, SIGNAL(imageIsUpdated(const ndarray<const GeoImage::raw_image_t,2>&)), m_wimage, SLOT(onImageIsUpdated(const ndarray<const GeoImage::raw_image_t,2>&)));

  m_wgt -> get_geotree() -> setItemSelected();
  m_geoimg -> setFirstImage();

  SetMsgLevel(level);
}

//--------------------------

void
GUIMain::showTips() 
{
  m_file_geo  -> setToolTip("Select \"geometry\" file");
  m_file_nda  -> setToolTip("Select ndarray with image file");
  m_but_exit  -> setToolTip("Exit application");
}

//--------------------------

void
GUIMain::setStyle()
{
  //m_file_geo->setFixedWidth(150);
  //m_file_nda->setFixedWidth(150);

  this -> setGeometry(0, 0, 500, 525);
  this -> setWindowTitle(tr("Detector alignment"));
  //this -> setContentsMargins(-9,-9,-9,-9);

  //this -> move(0,0);  

  //m_guilogger -> move(this->pos().x(), this->pos().y() + this->size().height());  

}

//--------------------------

void 
GUIMain::resizeEvent(QResizeEvent *event)
{
  stringstream ss; ss << "w:" << event->size().width() << " h:" <<  event->size().height();
  setWindowTitle(ss.str().c_str());
}

//--------------------------

void
GUIMain::moveEvent(QMoveEvent *event)
{
  stringstream ss; ss << "x:" << event->pos().x() << " y:" << event->pos().y();
  setWindowTitle(ss.str().c_str());
}

//--------------------------

void 
GUIMain::closeEvent(QCloseEvent *event)
{
  QWidget::closeEvent(event);
  stringstream ss; ss << "closeEvent(...): type = " << event -> type();
  MsgInLog(_name_(), INFO, ss.str());

  if(m_guiimv) m_guiimv -> close();
  if(m_guilogger) m_guilogger -> close();
  m_guiimv   = 0;
  m_guilogger = 0;

  SaveLog("work/z-log.txt", true);
}

//--------------------------

void 
GUIMain::mousePressEvent(QMouseEvent *event)
{
  //int x = event->pos().x();
  //int y = event->pos().y();
  //QString text = "mousePressEvent: " + QString::number(x) + "," + QString::number(y);
  //std::cout << text.toStdString()  << std::endl;
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------

void 
GUIMain::onButSave()
{
  MsgInLog(_name_(), DEBUG, "onButSave()");

  //char* search_fmt="*.data *.txt \n *"; 
  std::string path = getGeometryFileName(); // from QGUtils

  QString path_file = QFileDialog::getSaveFileName(this, tr("Select input file"), 
                                                   QString(path.c_str()), tr("*.data *.txt \n *"));
  std::string str_path_file = path_file.toStdString();

  if(str_path_file.empty()) {
    MsgInLog(_name_(), INFO, "Cancel file selection");
    return;
  }

  MsgInLog(_name_(), INFO, "Selected file name: " + str_path_file);
  m_wgt -> get_geotree() -> saveGeometryInFile(str_path_file);
}

//--------------------------

void 
GUIMain::onButExit()
{
  MsgInLog(_name_(), INFO, "onButExit"); 
  this->close(); // will call closeEvent(...)
}

//--------------------------

} // namespace PSQt

//--------------------------
