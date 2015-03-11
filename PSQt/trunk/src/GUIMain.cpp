//--------------------------

#include "PSQt/GUIMain.h"
#include "PSQt/Logger.h"
#include "PSQt/QGUtils.h"
#include "AppUtils/AppDataPath.h"

#include <sstream>   // for stringstream
#include <iostream>    // cout
#include <fstream>    // ifstream(fname), ofstream

//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

GUIMain::GUIMain( QWidget *parent
                , const LEVEL& level
		, const std::string& gfname
		, const std::string& ifname
   ): Frame(parent)
//  : QWidget(parent)
{
  MsgInLog(_name_(), INFO, "Create main control window"); 

  const std::string base_dir = "/reg/g/psdm/detector/alignment/cspad/calib-cxi-ds1-2014-05-15/";
  const std::string fname_geo = (gfname.empty()) ? (base_dir + "calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data") : gfname; 
  const std::string fname_nda = (ifname.empty()) ? (base_dir + "cspad-arr-cxid2714-r0023-lysozyme-rings.txt") : ifname; 

  //const std::string base_dir = "/reg/g/psdm/detector/alignment/cspad/calib-cxi-ds2-2015-01-20/";
  //const std::string fname_geo = base_dir + "calib/CsPad::CalibV1/CxiDs2.0:Cspad.0/geometry/0-end.data"; 
  //const std::string fname_nda = base_dir + "cspad-ndarr-max-cxig0715-r0023-lysozyme-rings.txt"; 

  const bool pbits = 0;

  m_guilogger = new PSQt::GUILogger(this, false, false); // true/false - show/do not show buttons, frame
  SetMsgLevel(level); // set level and update guilogger window for privious messages

  m_file_geo  = new PSQt::WdgFile(this, "Set geometry", fname_geo, "*.data \n *", false);
  m_file_nda  = new PSQt::WdgFile(this, "Set ndarray",  fname_nda, "*.txt *.dat \n *", false);
  m_wgt       = new PSQt::WdgGeoTree(this, fname_geo, pbits);
  m_wge       = new PSQt::WdgGeo(this);
  m_geoimg    = new PSQt::GeoImage(m_wgt->geoacc(), fname_nda);  
  m_guiimv    = new PSQt::GUIImageViewer(0);
  m_wimage    = m_guiimv->wdgImage();
 
  m_but_image = new QPushButton("Image", this);
  m_but_save  = new QPushButton("Save", this);

  m_bbox = new QHBoxLayout();
  m_bbox -> addWidget(m_but_save);
  m_bbox -> addStretch(1);
  m_bbox -> addWidget(m_but_image);

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

  m_vsplit = new QSplitter(Qt::Vertical); 
  m_vsplit -> addWidget(m_wmbox); 
  m_vsplit -> addWidget(m_guilogger); 

  m_vbox = new QVBoxLayout();
  m_vbox -> addWidget(m_vsplit);

  this -> setLayout(m_vbox);

  showTips();
  setStyle();

  connect(m_but_image, SIGNAL(clicked()), this, SLOT(onButImage()));
  connect(m_but_save,  SIGNAL(clicked()), this, SLOT(onButSave()));

  connect(m_wgt->get_view(), SIGNAL(selectedGO(shpGO&)),
          m_wge,             SLOT(setNewGO(shpGO&)));

  connect(m_file_geo,        SIGNAL(fileNameIsChanged(const std::string&)), 
          m_wgt->get_view(), SLOT(updateTreeModel(const std::string&))); 

  connect(m_wgt->get_view(), SIGNAL(geometryIsLoaded(PSCalib::GeometryAccess*)), 
          m_geoimg,          SLOT(onGeometryIsLoaded(PSCalib::GeometryAccess*))); 

  connect(m_file_nda, SIGNAL(fileNameIsChanged(const std::string&)), 
          m_geoimg,   SLOT(onImageFileNameIsChanged(const std::string&))); 

  connect(m_wge,      SIGNAL(geoIsChanged(shpGO&)), 
          m_geoimg,   SLOT(onGeoIsChanged(shpGO&)));

  connect(m_geoimg,   SIGNAL(imageIsUpdated(ndarray<GeoImage::raw_image_t,2>&)), 
          m_wimage,   SLOT(onImageIsUpdated(ndarray<GeoImage::raw_image_t,2>&)));

  connect(m_geoimg,                 SIGNAL(imageIsUpdated(ndarray<GeoImage::raw_image_t,2>&)), 
          m_guiimv->getImageProc(), SLOT(onImageIsUpdated(ndarray<GeoImage::raw_image_t,2>&)));

  // Complete initialization through connected signals & slots
  m_wgt -> get_geotree() -> setItemSelected();
  m_geoimg -> setFirstImage();
}

//--------------------------

void
GUIMain::showTips() 
{
  m_file_geo  -> setToolTip("Select geometry file");
  m_file_nda  -> setToolTip("Select ndarray with image file");
  m_but_image -> setToolTip("Open/close image window");
}

//--------------------------

void
GUIMain::setStyle()
{
  AppUtils::AppDataPath adp_icon_image("PSQt/icons/icon-monitor.png");
  AppUtils::AppDataPath adp_icon_exit("PSQt/icons/exit.png");
  AppUtils::AppDataPath adp_icon_save("PSQt/icons/save.png");

  m_but_image -> setIcon(QIcon(QString(adp_icon_image.path().c_str())));
  m_but_save  -> setIcon(QIcon(QString(adp_icon_save.path().c_str()))); 

  this -> setGeometry(0, 0, 700, 726);
  this -> setWindowTitle(tr("Detector alignment"));

  m_guiimv -> move(this->pos().x() + this->size().width() + 8, this->pos().y());  
  m_guiimv -> show();

  //m_file_geo->setFixedWidth(150);
  //m_file_nda->setFixedWidth(150);
  //this -> setContentsMargins(-9,-9,-9,-9);
  //this -> move(0,0);  
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

  SaveLog("work/z-log.txt", true); // true - insert time-stamp
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

void 
GUIMain::onButImage()
{
  m_guiimv -> setVisible(! (m_guiimv->isVisible()));
  m_guiimv -> move(this->pos().x() + this->size().width() + 8, this->pos().y());  
  stringstream ss; ss << "Image window " << ((m_guiimv->isVisible()) ? "is open" : "closed");
  MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------

} // namespace PSQt

//--------------------------
