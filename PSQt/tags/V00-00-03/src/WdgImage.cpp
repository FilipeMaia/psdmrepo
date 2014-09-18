//--------------------------
#include "PSQt/WdgImage.h"
#include "PSCalib/GeometryAccess.h"
#include "ndarray/ndarray.h" // for img_from_pixel_arrays(...)

#include "PSQt/QGUtils.h"

#include <iostream>    // for std::cout
#include <fstream>     // for std::ifstream(fname)
//using namespace std; // for cout without std::
//#include <math.h>  // atan2

namespace PSQt {

//--------------------------

WdgImage::WdgImage( QWidget *parent, const std::string& fname_geo, const std::string& fname_img )
  : QLabel(parent)
{
  m_geo_img = new GeoImage(fname_geo, fname_img);

  this -> setFrame();
  //this -> setText("Test text for this QLabel");
  //this -> setGeometry(200, 100, 500, 500);
  //this -> setWindowTitle("Image For Geometry");

  //this -> setAutoFillBackground (true); // MUST BE TRUE TO DRAW THE BACKGROUND COLOR SET IN Palette
  //this -> setMinimumHeight(200);
  this -> setMinimumSize(700,200);
  //this -> setPalette ( QPalette(QColor(255, 255, 255, 255)) );

  this -> setAlignment(Qt::AlignTop | Qt::AlignLeft);


  this -> setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);

  //////////////////////////////////
  //this -> setScaledContents (false);
  this -> setScaledContents (true);
  //////////////////////////////////

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

  m_pixmap_raw = 0;
  m_pixmap_scl = 0;

  m_painter = new QPainter(this);

  loadImageFromFile("/reg/neh/home1/dubrovin/LCLS/pubs/reflective-geometry.png");
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
WdgImage::zoomInImage()
{
  std::cout << "WdgImage::zoomInImage()\n";

  float sclx = float(m_pixmap_scl->size().width())  / this->size().width();  
  float scly = float(m_pixmap_scl->size().height()) / this->size().height();  

  //std::cout << "scale x:" << sclx << " y:"      << scly << '\n'; 

  int p1x = int( m_point1->x()*sclx );
  int p1y = int( m_point1->y()*scly );
  int p2x = int( m_point2->x()*sclx );
  int p2y = int( m_point2->y()*scly );

  int xmin =min(p1x, p2x);
  int xmax =max(p1x, p2x); 
  int ymin =min(p1y, p2y);
  int ymax =max(p1y, p2y);

  *m_pixmap_scl = m_pixmap_scl->copy(xmin, ymin, xmax-xmin, ymax-ymin);
  setPixmap(m_pixmap_scl->scaled(this->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
  //setPixmap(*m_pixmap_scl);
}

//--------------------------

void 
WdgImage::setPixmapScailedImage(const QImage* image)
{  
  //if image is available - reset m_pixmap_raw
  if(image) {
    if(m_pixmap_raw) {delete m_pixmap_raw; m_pixmap_raw=0;}
    m_pixmap_raw = new QPixmap(QPixmap::fromImage(*image));
  }

  m_pixmap_scl = new QPixmap(*m_pixmap_raw);
  setPixmap(m_pixmap_scl->scaled(this->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
}

//--------------------------

void
WdgImage::setFrame() 
{
  m_frame = new QFrame(this);
  m_frame -> setFrameStyle ( QFrame::Box | QFrame::Sunken); // or
  m_frame -> setLineWidth(0);
  m_frame -> setMidLineWidth(1);
  m_frame -> setCursor(Qt::SizeAllCursor); // Qt::WaitCursor, Qt::PointingHandCursor
  //m_frame -> setStyleSheet("background-color: rgb(0, 255, 255); color: rgb(255, 255, 100)");
  m_frame -> setGeometry(this->rect());
  m_frame -> setVisible(true);
}

//--------------------------

void 
WdgImage::resizeEvent(QResizeEvent *event)
{
  //m_frame -> setFrameRect (this->rect());
  m_frame->setGeometry(0, 0, event->size().width(), event->size().height());
  
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
  std::cout << "WdgImage::closeEvent(...): type = " << event -> type() << std::endl;
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

  m_is_pushed = false;

  if(e->button() == 4) { // for middle button
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
  std::cout << "WdgImage::loadImageFromFile: " << fname << '\n';
  //clear();

  QImage image(fname.c_str());
  setPixmapScailedImage(&image);
}

//--------------------------

void 
WdgImage::setColorPixmap()
{
  std::cout << "WdgImage::setColorPixmap()\n";
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
  std::cout << "WdgImage::setColorWhellPixmap()\n";
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
  std::cout << "WdgImage::setColorBar()\n";
  uint32_t* ctable = ColorTable(cols, hue1, hue2);
  uint32_t dimg[rows][cols]; 

  for(int r=0; r<rows; ++r) {
    std::memcpy(&dimg[r][0], &ctable[0], cols*sizeof(uint32_t));
    //dimg[r][c] = ctable[c];
  }

  QImage image((const uchar*) &dimg[0], cols, rows, QImage::Format_ARGB32);
  setPixmapScailedImage(&image);
}

//--------------------------

void 
WdgImage::setCameraImage()
{
  typedef PSCalib::GeometryAccess::image_t image_t;

  std::cout << "WdgImage::setCameraImage()\n";

  const ndarray<const image_t, 2> dnda = m_geo_img->get_image();

  ndarray<uint32_t, 2> inda(dnda.shape());

  // Define image_t to uint32_t conversion parameters
  image_t dmin=dnda[0][0];
  image_t dmax=dnda[0][0];
  ndarray<const image_t, 2>::iterator itd;
  for(itd=dnda.begin(); itd!=dnda.end(); ++itd) { 
    if( *itd < dmin ) dmin = *itd;
    if( *itd > dmax ) dmax = *itd;
  }
  image_t k = (dmax-dmin) ? 0xFFFFFF/(dmax-dmin) : 1; 

  std::cout << "     dnda: " << dnda
            << "\n   dmin: " << dmin
            << "\n   dmax: " << dmax
            << "\n      k: " << k
            << '\n';

  // Convert image_t to uint32_t Format_ARGB32
  ndarray<uint32_t, 2>::iterator iti;
  for(itd=dnda.begin(), iti=inda.begin(); itd!=dnda.end(); ++itd, ++iti) { 
    *iti = uint32_t( (*itd-dmin)*k ) + 0xFF000000; // converts to 24bits adds alpha layer
  }

  QImage image((const uchar*) &inda[0][0], inda.shape()[1], inda.shape()[0], QImage::Format_ARGB32);
  setPixmapScailedImage(&image);
}

//--------------------------

void 
WdgImage::onFileNameChanged(const std::string& fname)
{
  std::cout << "WdgImage::onFileNameChanged(string) - slot: fname = " << fname << '\n';  
  loadImageFromFile(fname);
}

//--------------------------

void 
WdgImage::onTest()
{
  std::cout << "WdgImage::onTest() - slot\n";  

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
