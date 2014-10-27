//--------------------------

#include "PSQt/LabColorRing.h"
#include "PSQt/QGUtils.h"

#include <math.h>    // atan2, abs, fmod
#include <iostream>    // cout

//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

LabColorRing::LabColorRing(QWidget *parent, const unsigned& ssize, float &h1, float &h2)
    : QLabel(parent)
    , m_parent(parent)
    , m_ssize(ssize)
    , m_h1(h1)
    , m_h2(h2)
    , m_poiC(float(m_ssize/2), float(m_ssize/2))
    , m_selected(0)
{
  //this -> setFrame();
  this -> setMargin(0);
  this -> setMinimumSize(m_ssize, m_ssize);
  this -> setAlignment(Qt::AlignTop | Qt::AlignLeft);
  //this -> setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);

  //////////////////////////////////
  //this -> setScaledContents (false);
  this -> setScaledContents (true);
  //////////////////////////////////

  showTips();
  setStyle();

  m_R = float(m_ssize/2); 
  m_frR1 = 0.4; 
  m_frR2 = 0.9; 
  m_R1 = m_R * m_frR1; 
  m_R2 = m_R * m_frR2; 

  m_rpicker = 10;

  m_pixmap_cring=0;
  setPens();
  setColorRing(512);

  this->setMouseTracking(true);
  //installEventFilter(this);
}

//--------------------------

void
LabColorRing::showTips() 
{
}

//--------------------------

void
LabColorRing::setFrame() 
{
  m_frame = new QFrame(this);
  m_frame -> setFrameStyle (QFrame::Box | QFrame::Sunken); // or
  //m_frame -> setFrameStyle ( QFrame::Box );    // NoFrame, Box, Panel, WinPanel, ..., StyledPanel 
  //m_frame -> setFrameShadow( QFrame::Sunken ); // Plain, Sunken, Raised 
  m_frame -> setLineWidth(0);
  m_frame -> setMidLineWidth(1);
  //m_frame -> setCursor(Qt::PointingHandCursor);     // Qt::WaitCursor, Qt::PointingHandCursor
  //m_frame -> setStyleSheet("background-color: rgb(0, 255, 255); color: rgb(255, 255, 100)");
}

//--------------------------

void
LabColorRing::setStyle() 
{
  this -> setFixedSize(m_ssize, m_ssize);
  this -> setCursor(Qt::ArrowCursor);     // Qt::WaitCursor, Qt::PointingHandCursor
  //this -> setCursor(Qt::PointingHandCursor);     // Qt::WaitCursor, Qt::PointingHandCursor
}

//--------------------------
//--------------------------

void
LabColorRing::setPens() 
{
  //QPen pen(Qt::black, 1, Qt::SolidLine); // Qt::DashLine
  //pen.setStyle(Qt::DashLine);
  //pen.setColor(QColor (200,200,200));

  m_pen_w1 = new QPen(Qt::black, 1, Qt::SolidLine);
  m_pen_w3 = new QPen(Qt::black, 3, Qt::SolidLine);

  QVector<qreal> dashes;
  qreal space = 4;
  dashes << 4 << space;

  m_pen1   = new QPen(Qt::black, 1, Qt::DashLine);
  m_pen2   = new QPen(Qt::white, 1, Qt::DashLine);
  m_pen1->setDashPattern(dashes);
  m_pen2->setDashPattern(dashes);
  m_pen2->setDashOffset(4);
}

//--------------------------

void 
LabColorRing::paintEvent(QPaintEvent *event)
{
  //QPainter painter(m_lab_cring);
  static unsigned count=0; count++;
  QLabel::paintEvent(event);

  //m_painter = new QPainter(this);

  m_painter.begin(this);
  //-----------
  setPoints();

  drawLines();
  drawCircs();
  //-----------
  m_painter.end();

  //std::cout << "WdgImage::paintEvent counter = " << count << '\n';
}

//--------------------------

void 
LabColorRing::setPoints()
{
  float h1 = DEG2RAD*m_h1;
  float h2 = DEG2RAD*m_h2;
  float s1 = sin(h1);
  float s2 = sin(h2);
  float c1 = cos(h1);
  float c2 = cos(h2);

  m_poi1 .setX( m_poiC.x() + m_R1 * c1 );
  m_poi1 .setY( m_poiC.y() - m_R1 * s1 );
  m_poi1e.setX( m_poiC.x() + m_R2 * c1 );
  m_poi1e.setY( m_poiC.y() - m_R2 * s1 );
  m_poi2 .setX( m_poiC.x() + m_R2 * c2 );
  m_poi2 .setY( m_poiC.y() - m_R2 * s2 );
  m_poi2e.setX( m_poiC.x() + m_R1 * c2 );
  m_poi2e.setY( m_poiC.y() - m_R1 * s2 );
}

//--------------------------

void 
LabColorRing::drawLines()
{
  m_painter.setPen(*m_pen1);
  m_painter.drawLine(m_poi1, m_poi1e);
  m_painter.drawLine(m_poi2, m_poi2e);

  m_painter.setPen(*m_pen2);
  m_painter.drawLine(m_poi1, m_poi1e);
  m_painter.drawLine(m_poi2, m_poi2e);

  float rm = m_rpicker;
  int   rs = int(m_rpicker * 2);
  //m_painter.setBrush(Qt::Dense6Pattern);
  m_painter.setPen(*m_pen_w3);
  //m_painter.drawEllipse(m_poi1, rm, rm);
  //m_painter.drawEllipse(m_poi2, rm, rm);  
  m_painter.drawRect(int(m_poi1.x()-rm), int(m_poi1.y()-rm), rs, rs );
  m_painter.drawRect(int(m_poi2.x()-rm), int(m_poi2.y()-rm), rs, rs );

  m_painter.drawText(m_poi1+QPoint(-4,4), QString(QChar(0x25cf)));
  m_painter.drawText(m_poi2+QPoint(-4,4), QString(QChar(0x25cf)));
}

//--------------------------

void 
LabColorRing::drawCircs()
{
  m_painter.setPen(*m_pen_w1);
  m_painter.drawEllipse(m_poiC, m_R1, m_R1);
  m_painter.drawEllipse(m_poiC, m_R2, m_R2);
}

//--------------------------

void 
LabColorRing::resizeEvent(QResizeEvent *event)
{
//  m_frame->setGeometry(0, 0, event->size().width(), event->size().height());
  //setWindowTitle("Window is resized");

  //this->setPixmap(m_pixmap_cring->scaled(this->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
}

//--------------------------

void 
LabColorRing::closeEvent(QCloseEvent *event)
{
  QWidget::closeEvent(event);
  //std::cout << "LabColorRing::closeEvent(...): type = " << event -> type() << std::endl;
  //MsgLog("LabColorRing", info, "closeEvent(...): type = " << event -> type());
}

//--------------------------

void
LabColorRing::moveEvent(QMoveEvent *event)
{
  //int x = event->pos().x();
  //int y = event->pos().y();
  //QString text = QString::number(x) + "," + QString::number(y);
  //setWindowTitle(text);
}

//--------------------------

void 
LabColorRing::mousePressEvent(QMouseEvent *e)
{
  /*
  std::cout << "mousePressEvent:"
            << "  button: " << e->button()
            << "  x(),y() = " << e->x() << ", " << e->y()
            << "  isActiveWindow(): " << this->isActiveWindow()
            << '\n';
  */

  if( (e->pos()-m_poi1).manhattanLength() < 2*m_rpicker ) { 
    m_selected = 1; 
    m_ang = m_h1; 
  }
  else if( (e->pos()-m_poi2).manhattanLength() < 2*m_rpicker ) { 
    m_selected = 2; 
    m_ang = m_h2; 
  }
  else {
    m_selected = 0; 
    return;
  }

  m_n360 = int(m_ang/360)*360;
  if (m_ang<0) m_n360-=360;
  m_ang_old = m_ang-m_n360;
  //m_ang_old = fmod(m_ang,360);

  //std::cout << "mousePressEvent: selected = " << m_selected << '\n';

  //QApplication::setOverrideCursor(Qt::ClosedHandCursor);
  //m_parent -> setCursor(Qt::ClosedHandCursor);
  this -> setCursor(Qt::ClosedHandCursor);
}

//--------------------------

void 
LabColorRing::mouseMoveEvent(QMouseEvent *e)
{
  //std::cout << "LabColorRing::mouseMoveEvent: "
  //          << "  x(),y() = "  << e->x() << ", " << e->y()
  //          << '\n';

  if(! m_selected) {
    if((e->pos()-m_poi1).manhattanLength() < 2*m_rpicker 
    || (e->pos()-m_poi2).manhattanLength() < 2*m_rpicker ) this -> setCursor(Qt::OpenHandCursor);
    else this -> unsetCursor(); // this->setCursor(Qt::ArrowCursor);
  }

  setHueAngle(e);
}

//--------------------------

void 
LabColorRing::mouseReleaseEvent(QMouseEvent *e)
{
  //std::cout << "mouseReleaseEvent:"
  //          << "  button: " << e->button()
  //          << "  x(),y() = " << e->x() << ", " << e->y()
  //          << '\n';

  //QApplication::restoreOverrideCursor();
  this->setCursor(Qt::ArrowCursor);

  setHueAngle(e);

  m_selected = 0;
}

//--------------------------

void 
LabColorRing::enterEvent(QEvent *e)
{
  //std::cout << "LabColorRing::enterEvent(.)\n";
}

//--------------------------

void 
LabColorRing::leaveEvent(QEvent *e)
{
  //std::cout << "LabColorRing::leaveEvent(.)\n";
}

//--------------------------

//bool 
//LabColorRing::eventFilter(QObject *obj, QEvent *e)
//{
//  //if(obj==this && (e->type()==QEvent::Enter || e->type()==QEvent::Leave)) 
//  std::cout << "LabColorRing::eventFilter(...) type: " << e->type()<< '\n';
//  if(obj!=this) return true;
//  if(e->type()==QEvent::MouseMove) return false; 
//  if(e->type()==QEvent::MouseButtonPress) return false; 
//  if(e->type()==QEvent::MouseButtonRelease) return false; 
//  if(e->type()==QEvent::Paint) return false; 
//  return true;
//}

//--------------------------

void 
LabColorRing::setHueAngle(QMouseEvent *e)
{
  if( !m_selected ) return;

  QPointF dc(e->pos() - m_poiC);

  float m_ang = atan2(-dc.y(),dc.x()) * RAD2DEG;
  m_ang = (m_ang<0) ? m_ang+360 : m_ang; // [0,360)

  if(m_ang_old>330 && m_ang<30)  m_n360 += 360;
  if(m_ang_old<30  && m_ang>330) m_n360 -= 360;

  if     ( m_selected == 1 ) m_h1 = m_ang + m_n360;
  else if( m_selected == 2 ) m_h2 = m_ang + m_n360;
  else return;    

  m_ang_old = m_ang;

  update();

  if(m_selected) {
    //std::cout << "LabColorRing::mouseReleaseEvent: emit signal hueAngleIsChanged()\n";
    emit hueAngleIsChanged(m_selected);
  }

}

//--------------------------
//--------------------------

void 
LabColorRing::onSetShifter(const unsigned& edited)
{
  //std::cout << "LabColorRing::onSetShifter() h1:" << m_h1 << " h2:" << m_h2 
  //          << " edited:" << edited << '\n';
  update();
}

//--------------------------

void 
LabColorRing::onButExit()
{
  //std::cout << "LabColorRing::onButExit()\n";
  this->close(); // will call closeEvent(...)
}

//--------------------------
/*
void 
LabColorRing::setPixmapForLabel(const QImage& image, QPixmap*& pixmap, QLabel*& label)
{
  if(pixmap) delete pixmap;
  pixmap = new QPixmap(QPixmap::fromImage(image));
  //else pixmap -> loadFromData ( (const uchar*) &dimg[0], unsigned(rows*cols) );

  label->setPixmap(pixmap->scaled(label->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
  //label->setPixmap(*pixmap);
}
*/
//--------------------------

void 
LabColorRing::setColorRing(const int& ssize)
{
  //std::cout << "LabColorRing::setColorRing() for ssize = " << ssize << '\n';
  //clear();

  const int isize = ssize*ssize;

  const int xc = ssize/2;
  const int yc = ssize/2;  
  const int rr = int(ssize*0.5);  

  uint32_t dimg[ssize][ssize]; 
  //std::fill_n(&dimg[0][0], int(isize), uint32_t(0xFF000000));
  std::fill_n(&dimg[0][0], int(isize), uint32_t(0xFFFFFFFF));

  for(int i=0; i<ssize; ++i) {
  for(int j=0; j<ssize; ++j) {

    float y = yc-i; // reflected
    float x = j-xc;
    float r = sqrt(x*x + y*y) / rr;

    float hue = atan2(y,x) * RAD2DEG;
    hue = (hue<0) ? hue+360 : hue;

    //if(0.5<r && r<1) dimg[i][j] = HSV2RGBA(hue, 1, r);
    if(m_frR1<r && r<m_frR2) dimg[i][j] = HSV2RGBA(hue, 1, 1);
  }
  }

  QImage image((const uchar*) &dimg[0], ssize, ssize, QImage::Format_ARGB32);
  //setPixmapForLabel(image, m_pixmap_cring, this);

  if(m_pixmap_cring) delete m_pixmap_cring;
  m_pixmap_cring = new QPixmap(QPixmap::fromImage(image));
  //else pixmap -> loadFromData ( (const uchar*) &dimg[0], unsigned(rows*cols) );

  this->setPixmap(m_pixmap_cring->scaled(this->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
}

//--------------------------

//--------------------------

//--------------------------

} // namespace PSQt

//--------------------------
