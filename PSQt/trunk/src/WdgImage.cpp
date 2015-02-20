//--------------------------
#include "PSQt/WdgImage.h"
#include "PSQt/Logger.h"
#include "PSCalib/GeometryAccess.h"
#include "ndarray/ndarray.h" // for img_from_pixel_arrays(...)

#include "PSQt/QGUtils.h"

#include <iostream>    // for std::cout
#include <fstream>     // for std::ifstream(fname)
//using namespace std; // for cout without std::
//#include <math.h>  // atan2
#include <cstring> // for memcpy

namespace PSQt {

//--------------------------

WdgImage::WdgImage(QWidget *parent, const std::string& ifname)
  : QLabel(parent)
  , m_frame(0)
  , m_painter(0)
  , m_geo_img(0)
  , m_pixmap_raw(0)
  , m_pixmap_scl(0)
  , m_pen1(0)  
  , m_pen2(0)  
  , m_point1(0)
  , m_point2(0)
  , m_rect1(0) 
  , m_rect2(0) 
{
  setWdgParams();

  const std::string fname = (ifname!=std::string()) ? ifname
                          : "/reg/neh/home1/dubrovin/LCLS/pubs/galaxy.jpeg";
  //                          : "/reg/neh/home1/dubrovin/LCLS/pubs/reflective-geometry.png";
  loadImageFromFile(fname);
}

//--------------------------

WdgImage::WdgImage( QWidget *parent, const QImage* image)
  : QLabel(parent)
  , m_frame(0)
  , m_painter(0)
  , m_geo_img(0)
  , m_pixmap_raw(0)
  , m_pixmap_scl(0)
  , m_pen1(0)  
  , m_pen2(0)  
  , m_point1(0)
  , m_point2(0)
  , m_rect1(0) 
  , m_rect2(0) 
{
  setWdgParams();
  setPixmapScailedImage(image);
}

//--------------------------

WdgImage::~WdgImage()
{
  if (m_frame)      delete m_frame;  
  if (m_painter)    delete m_painter;  
  if (m_pen1)       delete m_pen1;  
  if (m_pen2)       delete m_pen2;  
  if (m_point1)     delete m_point1;  
  if (m_point2)     delete m_point2;  
  if (m_rect1)      delete m_rect1;  
  if (m_rect2)      delete m_rect2;  
  if (m_pixmap_raw) delete m_pixmap_raw;  
  if (m_pixmap_scl) delete m_pixmap_scl;  
  if (m_geo_img)    delete m_geo_img;  
}

//--------------------------

void 
WdgImage::setWdgParams()
{
  //this -> setFrame();
  //this -> setText("Test text for this QLabel");
  //this -> setGeometry(200, 100, 500, 500);
  //this -> setWindowTitle("Image For Geometry");

  //this -> setAutoFillBackground (true); // MUST BE TRUE TO DRAW THE BACKGROUND COLOR SET IN Palette
  //this -> setMinimumHeight(200);
  this -> setMinimumSize(606,606);
  //this -> setPalette ( QPalette(QColor(255, 255, 255, 255)) );

  this -> setAlignment(Qt::AlignTop | Qt::AlignLeft);


  this -> setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);

  //////////////////////////////////
  //this -> setScaledContents (false);
  this -> setScaledContents (true);
  //////////////////////////////////

  this -> setCursor(Qt::PointingHandCursor); // Qt::SizeAllCursor, Qt::WaitCursor, Qt::PointingHandCursor

  //this -> setIndent(100);
  this -> setMargin(0);

  //this -> installEventFilter(this);

  QVector<qreal> dashes;
  qreal space = 4;
  dashes << 4 << space;

  m_pen1   = new QPen(Qt::black, 1, Qt::DashLine);
  m_pen2   = new QPen(Qt::white, 1, Qt::DashLine);
  m_pen1->setDashPattern(dashes);
  m_pen2->setDashPattern(dashes);
  m_pen2->setDashOffset(4);
  m_point1 = new QPoint();
  m_point2 = new QPoint();
  m_rect1  = new QRect();
  m_rect2  = new QRect();
  m_is_pushed = false;

  //std::cout << "Point A\n";
  m_painter = new QPainter();
  //m_painter = new QPainter(this);
  //std::cout << "Point B\n";

  m_ncolors = 1024; // =0 - color table is not used
  m_hue1    = -120;
  m_hue2    = -360;

  this->resetZoom();
}

//--------------------------

void 
WdgImage::paintEvent(QPaintEvent *event)
{
  static unsigned count=0; count++;
  QLabel::paintEvent(event);

  m_painter->begin(this);

  //-----------
  //drawLine();
  if(m_is_pushed) drawRect();
  //-----------

  m_painter->end();

  //std::cout << "WdgImage::paintEvent counter = " << count << '\n';
}

//--------------------------

void 
WdgImage::drawLine()
{
  QPen pen(Qt::black, 2, Qt::SolidLine);
  m_painter->setPen(pen);
  m_painter->drawLine(20, 20, 250, 20);

  pen.setStyle(Qt::DashLine);
  m_painter->setPen(pen);
  m_painter->drawLine(20, 40, 250, 40);
}

//--------------------------

void 
WdgImage::drawRect()
{
  m_rect1->setCoords(m_point1->x(), m_point1->y(), m_point2->x(), m_point2->y());
  m_rect2->setCoords(m_point1->x(), m_point1->y(), m_point2->x(), m_point2->y());
  m_painter->setPen  (*m_pen1);
  m_painter->drawRect(*m_rect1); 
  m_painter->setPen  (*m_pen2);
  m_painter->drawRect(*m_rect2); 
}

//--------------------------

void 
WdgImage::resetZoom()
{
  m_point1->setX(0);
  m_point1->setY(0);
  m_xmin_raw = 0;
  m_ymin_raw = 0;
  m_zoom_is_on = false;
}

//--------------------------

void 
WdgImage::zoomInImage()
{
  MsgInLog(_name_(), INFO, "zoomInImage()");

  //std::cout << "  x1:" << m_point1->x() << "  y1:" << m_point1->y() 
  //          << "  x2:" << m_point2->x() << "  y2:" << m_point2->y()<< '\n'; 

  if(m_point1->x() != 0 and m_point1->y() != 0) {

    float sclx = float(m_pixmap_scl->size().width())  / this->size().width();  
    float scly = float(m_pixmap_scl->size().height()) / this->size().height();  
    
    int p1x = int( m_point1->x()*sclx );
    int p1y = int( m_point1->y()*scly );
    int p2x = int( m_point2->x()*sclx );
    int p2y = int( m_point2->y()*scly );
    
    int xmin = min(p1x, p2x);
    int xmax = max(p1x, p2x); 
    int ymin = min(p1y, p2y);
    int ymax = max(p1y, p2y);
    
    m_xmin_raw += xmin;
    m_ymin_raw += ymin;
    m_xmax_raw = m_xmin_raw + xmax-xmin;
    m_ymax_raw = m_ymin_raw + ymax-ymin;
    
    m_point1->setX(0);
    m_point1->setY(0);

    m_zoom_is_on = true;
  }

  if (m_zoom_is_on) {
     *m_pixmap_scl = m_pixmap_raw->copy(m_xmin_raw, m_ymin_raw, m_xmax_raw-m_xmin_raw, m_ymax_raw-m_ymin_raw);
     setPixmap(m_pixmap_scl->scaled(this->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
  }
}

//--------------------------

void 
WdgImage::setPixmapScailedImage(const QImage* image)
{  
  //if image is available - reset m_pixmap_raw
  if(image) {
    if(m_pixmap_raw) delete m_pixmap_raw;
    m_pixmap_raw = new QPixmap(QPixmap::fromImage(*image));
  }

  if (m_zoom_is_on) 
    zoomInImage();

  else {
    if (m_pixmap_scl) delete m_pixmap_scl;
    m_pixmap_scl = new QPixmap(*m_pixmap_raw);
    setPixmap(m_pixmap_scl->scaled(this->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
  }
}

//--------------------------

void
WdgImage::setFrame() 
{
  m_frame = new QFrame(this);
  m_frame -> setFrameStyle ( QFrame::Box | QFrame::Sunken); // or
  m_frame -> setLineWidth(0);
  m_frame -> setMidLineWidth(1);
  //m_frame -> setCursor(Qt::OpenHandCursor); // Qt::SizeAllCursor, Qt::WaitCursor, Qt::PointingHandCursor
  //m_frame -> setStyleSheet("background-color: rgb(0, 255, 255); color: rgb(255, 255, 100)");
  m_frame -> setGeometry(this->rect());
  m_frame -> setVisible(true);
}

//--------------------------

void 
WdgImage::resizeEvent(QResizeEvent *event)
{
  //m_frame->setGeometry(0, 0, event->size().width(), event->size().height());
  
  //std::cout << "WdgImage::resizeEvent(...): w=" << event->size().width() 
  //	    << "  h=" << event->size().height() << '\n';

  setPixmap(m_pixmap_scl->scaled(this->size(), Qt::KeepAspectRatio, Qt::FastTransformation));

  setWindowTitle("Window is resized");
}

//--------------------------

void 
WdgImage::closeEvent(QCloseEvent *event)
{
  QWidget::closeEvent(event);
  stringstream ss; ss << "closeEvent(...): type = " << event -> type();
  MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------

void 
WdgImage::mousePressEvent(QMouseEvent *e)
{
  /*
  std::cout << "mousePressEvent:"
            << "  button: " << e->button()
            << "  x(),y() = " << e->x() << ", " << e->y()
            << "  isActiveWindow(): " << this->isActiveWindow()
            << '\n';
  */
  this -> setCursor(Qt::ClosedHandCursor);

  m_point1->setX(e->x());
  m_point1->setY(e->y());
  m_point2->setX(e->x());
  m_point2->setY(e->y());

  m_is_pushed = true;
  update();
}

//--------------------------

void 
WdgImage::mouseReleaseEvent(QMouseEvent *e)
{
  /*
  std::cout << "mouseReleaseEvent:"
            << "  button: " << e->button()
            << "  x(),y() = " << e->x() << ", " << e->y()
            << '\n';
  */

  //this -> unsetCursor(); 
  this -> setCursor(Qt::PointingHandCursor); // Qt::SizeAllCursor, Qt::WaitCursor, Qt::PointingHandCursor

  m_is_pushed = false;

  if(e->button() == 4) { // for middle button

    this->resetZoom();
    setPixmapScailedImage();
    update();
    return;
  }

  m_point2->setX(e->x());
  m_point2->setY(e->y());

  QPoint dist = *m_point2 - *m_point1;
  if(this->rect().contains(*m_point2) && dist.manhattanLength() > 5) zoomInImage();

  update();
}

//--------------------------

void 
WdgImage::mouseMoveEvent(QMouseEvent *e)
{
  //std::cout << "mouseMoveEvent: "
  //          << "  x(),y() = "  << e->x() << ", " << e->y()
  //          << '\n';
  m_point2->setX(e->x());
  m_point2->setY(e->y());
  update();
}

//--------------------------

void 
WdgImage::loadImageFromFile(const std::string& fname)
{
  MsgInLog(_name_(), INFO, "Load image from file " + fname);
  //std::cout << "WdgImage::loadImageFromFile: " << fname << '\n';
  //clear();

  QImage image(fname.c_str());

  this->resetZoom();
  setPixmapScailedImage(&image);
}

//--------------------------

void 
WdgImage::onFileNameChanged(const std::string& fname)
{
  MsgInLog(_name_(), INFO, "onFileNameChanged: " + fname);
  //std::cout << _name_() << "onFileNameChanged(string) - slot: fname = " << fname << '\n';  
  loadImageFromFile(fname);
}

//--------------------------
void 
WdgImage::onImageIsUpdated(const ndarray<const GeoImage::raw_image_t,2>& nda)
{
  stringstream ss; ss << "onImageIsUpdated(): Receive and update raw image in window, rows:" << nda.shape()[0] << " cols:" << nda.shape()[1] ;
  MsgInLog(_name_(), INFO, ss.str()); 

  const ndarray<GeoImage::image_t,2> nda_norm = 
        getUint32NormalizedImage<const GeoImage::raw_image_t>(nda, m_ncolors, m_hue1, m_hue2); // from QGUtils
  
  onNormImageIsUpdated(nda_norm);
  update();
}

//--------------------------
void 
WdgImage::onNormImageIsUpdated(const ndarray<GeoImage::image_t,2>& nda)
{
  const unsigned int rows = nda.shape()[0];
  const unsigned int cols = nda.shape()[1];

  stringstream ss; ss << "onNormImageIsUpdated::Receive and update normalized image in window, rows:" << nda.shape()[0] << " cols:" << nda.shape()[1] ;
  MsgInLog(_name_(), DEBUG, ss.str()); 
  
  QImage image((const uchar*) nda.data(), cols, rows, QImage::Format_ARGB32);
  setPixmapScailedImage(&image);
  
  static unsigned counter=0; stringstream sst; sst << "Image # " << ++counter;
  setWindowTitle(sst.str().c_str());
}

//--------------------------
//--------------------------
//----   Test images   -----
//--------------------------
//--------------------------

void 
WdgImage::setColorPixmap()
{
  MsgInLog(_name_(), INFO, "setColorPixmap()");
  //std::cout << "WdgImage::setColorPixmap()\n";
  //clear();

  const int ssize = 1024;
  const int isize = ssize*ssize;

  uint32_t dimg[ssize][ssize]; 
  std::fill_n(&dimg[0][0], int(isize), uint32_t(0xFF000000));

  int vRx = 512 - 128;
  int vRy = 512 - 128;
  int vGx = 512 + 128;
  int vGy = 512 - 128;
  int vBx = 512;
  int vBy = 512 - 128 + 222;

  for(int i=0; i<ssize; ++i) {
  for(int j=0; j<ssize; ++j) {

    int rR = (int) sqrt( pow(i-vRx,2) + pow(j-vRy,2) );
    int rG = (int) sqrt( pow(i-vGx,2) + pow(j-vGy,2) );
    int rB = (int) sqrt( pow(i-vBx,2) + pow(j-vBy,2) );

    //int	r = 255-rR; r = (r>0) ? r : 0;
    //int	g = 255-rG; g = (g>0) ? g : 0;
    //int	b = 255-rB; b = (b>0) ? b : 0;
    int	r = (rR<255) ? rR : 255;
    int	g = (rG<255) ? rG : 255;
    int	b = (rB<255) ? rB : 255;

    dimg[i][j] += (r<<16) + (g<<8) + b;
  }
  }

  QImage image((const uchar*) &dimg[0], ssize, ssize, QImage::Format_ARGB32);
  setPixmapScailedImage(&image);
}

//--------------------------

void 
WdgImage::setColorWhellPixmap()
{
  MsgInLog(_name_(), INFO, "setColorWhellPixmap()");
  //std::cout << "WdgImage::setColorWhellPixmap()\n";
  //clear();

  const int ssize = 1024;
  const int isize = ssize*ssize;

  const int xc = 512;
  const int yc = 512;  

  const float RAD2DEG =  180/3.14159265;

  uint32_t dimg[ssize][ssize]; 
  std::fill_n(&dimg[0][0], int(isize), uint32_t(0xFF000000));


  for(int i=0; i<ssize; ++i) {
  for(int j=0; j<ssize; ++j) {

    float x = i-xc;
    float y = j-yc;
    float r = sqrt(x*x + y*y) / 512;
    r = (r<1) ? r : 0;

    float hue = atan2(y,x) * RAD2DEG;
    hue = (hue<0) ? hue+360 : hue;

    dimg[i][j] = HSV2RGBA(hue, 1, r);
  }
  }

  QImage image((const uchar*) &dimg[0], ssize, ssize, QImage::Format_ARGB32);
  setPixmapScailedImage(&image);
}

//--------------------------

void
WdgImage::setColorBar( const unsigned& rows, 
                       const unsigned& cols,
                       const float&    hue1,
                       const float&    hue2
                      )
{
  MsgInLog(_name_(), INFO, "setColorBar()");
  //std::cout << "WdgImage::setColorBar()\n";
  uint32_t* ctable = ColorTable(cols, hue1, hue2);
  uint32_t dimg[rows][cols]; 

  for(unsigned r=0; r<rows; ++r) {
    std::memcpy(&dimg[r][0], &ctable[0], cols*sizeof(uint32_t));
    //dimg[r][c] = ctable[c];
  }

  QImage image((const uchar*) &dimg[0], cols, rows, QImage::Format_ARGB32);
  setPixmapScailedImage(&image);
}

//--------------------------

void 
WdgImage::setCameraImage(const std::string& ifname_geo, const std::string& ifname_img)
{
  typedef PSCalib::GeometryAccess::image_t image_t;

  const std::string base_dir = "/reg/g/psdm/detector/alignment/cspad/calib-cxi-ds1-2014-05-15/";
  const std::string fname_geo = (ifname_geo != std::string()) ? ifname_geo
                              : base_dir + "calib/CsPad::CalibV1/CxiDs1.0:Cspad.0/geometry/2-end.data"; 
  const std::string fname_img = (ifname_img != std::string()) ? ifname_img
                              : base_dir + "cspad-arr-cxid2714-r0023-lysozyme-rings.txt"; 

  MsgInLog(_name_(), INFO, "setCameraImage()");

  if (m_geo_img) delete m_geo_img;  
  m_geo_img = new GeoImage(fname_geo, fname_img);

  ndarray<uint32_t, 2> inda = m_geo_img->getNormalizedImage();

  QImage image((const uchar*) &inda[0][0], inda.shape()[1], inda.shape()[0], QImage::Format_ARGB32);
  setPixmapScailedImage(&image);
}

//--------------------------

void 
WdgImage::onTest()
{
  MsgInLog(_name_(), INFO, "onTest() - slot");
  //std::cout << "WdgImage::onTest() - slot\n";  

  static unsigned counter = 0; ++counter;

  if(counter%6 == 1) setColorWhellPixmap();
  if(counter%6 == 2) setColorBar();
  if(counter%6 == 3) setColorPixmap();
  if(counter%6 == 4) setCameraImage();
  if(counter%6 == 5) this -> setScaledContents (false);
  if(counter%6 == 0) this -> setScaledContents (true);
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------

} // namespace PSQt

//--------------------------
