//--------------------------

#include "PSQt/WdgColorBar.h"
#include "PSQt/Logger.h"

#include <QString>
#include <string>

#include <iostream>  // cout
#include <cstring>   // for memcpy, placed in the std namespace
#include <sstream>   // for stringstream
#include <algorithm> // for fill_n
#include <iomanip>   // for setw, setfill

//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

WdgColorBar::WdgColorBar( QWidget *parent, 
                          const float& h1, 
                          const float& h2, 
                          const unsigned& colors, 
                          const ORIENT& orient, 
                          const float& aspect)
    : QLabel(parent)
    , m_lab_cbar(this)
    , m_pixmap_cbar(0)
{
  //showTips();
  setStyle();

  setColorBar(h1, h2, colors, orient, aspect);

  //connectForTest(); 
}

//--------------------------

void WdgColorBar::connectForTest() 
{
  // connections for internal test 
  connect(this, SIGNAL(pressColorBar(QMouseEvent*, const float&)),
          this, SLOT(testPressColorBar(QMouseEvent*, const float&)));

  connect(this, SIGNAL(releaseColorBar(QMouseEvent*, const float&)),
          this, SLOT(testReleaseColorBar(QMouseEvent*, const float&)));

  connect(this, SIGNAL(moveOnColorBar(QMouseEvent*, const float&)),
          this, SLOT(testMoveOnColorBar(QMouseEvent*, const float&)));
}

//--------------------------

WdgColorBar::~WdgColorBar()
{
  delete m_pixmap_cbar;
}

//--------------------------

void
WdgColorBar::showTips() 
{
  this -> setToolTip("Color bar with mouse button control");
}

//--------------------------

void
WdgColorBar::setStyle() 
{
  this -> setMargin(0);
  this -> setWindowTitle(tr("Color bar"));
  //this -> move(100,50);  
  //this -> setMinimumSize(m_length, m_width);
  //this -> setFixedSize(m_length, m_width);
  //this -> setFixedWidth(m_width+22);
}

//--------------------------

void 
WdgColorBar::resizeEvent(QResizeEvent *e)
{
  this->setPixmap(m_pixmap_cbar->scaled(this->size(), Qt::IgnoreAspectRatio, Qt::FastTransformation));
  //this->setPixmap(m_pixmap_cbar->scaled(this->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
  //std::stringstream ss; ss << _name_() << " resized: w=" << e->size().width() 
  //                                              << " h=" << e->size().height(); 
  //MsgInLog(_name_(), INFO, ss.str());
  //setWindowTitle(ss.str().c_str());
  //std::cout << ss.str() << '\n';
}

//--------------------------

void 
WdgColorBar::closeEvent(QCloseEvent *e)
{
  QWidget::closeEvent(e);
  //std::stringstream ss; ss << "closeEvent(...): type = " << e->type();
  //MsgInLog(_name_(), INFO, ss.str());
}

//--------------------------
void
WdgColorBar::moveEvent(QMoveEvent *e)
{
  //int x = e->pos().x();
  //int y = e->pos().y();
  //QString text = QString::number(x) + "," + QString::number(y);
  //setWindowTitle(text);
}

//--------------------------

float
WdgColorBar::fractionOfColorBar(QMouseEvent *e)
{
  //QMouseButton button = e->button()
  float fraction = 0;

  float width  = this->size().width();
  float height = this->size().height();
  float x = e->pos().x();
  float y = e->pos().y();

  if      (m_orient == HR) fraction = (width)  ? x/width : 0;
  else if (m_orient == HL) fraction = (width)  ? (width-x)/width : 0;
  else if (m_orient == VU) fraction = (height) ? (height-y)/height : 0;
  else if (m_orient == VD) fraction = (height) ? y/height : 0;
  
  return fraction;
}

//--------------------------

void 
WdgColorBar::mousePressEvent(QMouseEvent *e)
{
  float f = fractionOfColorBar(e);
  emit pressColorBar(e, f);
}

//--------------------------

void 
WdgColorBar::mouseReleaseEvent(QMouseEvent *e)
{
  float f = fractionOfColorBar(e);
  emit releaseColorBar(e, f);
}

//--------------------------

void 
WdgColorBar::mouseMoveEvent(QMouseEvent *e)
{
  float f = fractionOfColorBar(e);
  emit moveOnColorBar(e, f);
}

//--------------------------

void 
WdgColorBar::onHueAnglesUpdated(const float& h1, const float& h2)
{
  this->setColorBar(h1, h2, m_colors, m_orient, m_aspect);

  this->setPixmap(m_pixmap_cbar->scaled(this->size(), Qt::IgnoreAspectRatio, Qt::FastTransformation));
  
  std::stringstream ss; ss << "Set hue angles h1:" << m_h1 << " h2:" << m_h2;
  MsgInLog(_name_(), DEBUG, ss.str());  
}

//--------------------------

//void 
//WdgColorBar::onButExit()
//{
//  //std::cout << "WdgColorBar::onButExit()\n";
//  this->close(); // will call closeEvent(...)
//}

//--------------------------

void
WdgColorBar::setColorBar( const float&    h1,
                          const float&    h2,
                          const unsigned& colors, 
                          const ORIENT&   orient, 
                          const float&    aspect
                        )
{
  m_h1     = h1;
  m_h2     = h2;
  m_colors = colors;
  m_orient = orient;
  m_aspect = aspect;

  const unsigned cols = (orient==HR || orient==HL) ? colors : unsigned(colors * aspect);
  const unsigned rows = (orient==HR || orient==HL) ? unsigned(colors * aspect) : colors;

  uint32_t dimg[rows][cols];
  uint32_t* ctable = ColorTable(colors, h1, h2);
  uint32_t ctable_mirror[colors]; 

  switch (orient) {
    default :   
    case HR :   
      for(unsigned r=0; r<rows; ++r)
        std::memcpy(&dimg[r][0], &ctable[0], cols*sizeof(uint32_t));
      break;
    
    case HL :   
      for(unsigned c=0; c<colors; ++c) ctable_mirror[c] = ctable[colors-c-1];
      for(unsigned r=0; r<rows; ++r)
        std::memcpy(&dimg[r][0], &ctable_mirror[0], cols*sizeof(uint32_t));
      break;
    
    case VU :
      for(unsigned r=0; r<rows; ++r) 
        std::fill_n(&dimg[r][0], int(cols), ctable[rows-r-1]);
      break;
    
    case VD :
      for(unsigned r=0; r<rows; ++r) 
        std::fill_n(&dimg[r][0], int(cols), ctable[r]);
      break;
    
    MsgInLog(_name_(), WARNING, "Wrong orientation parameter for color bar object.");  
  }

  QImage image((const uchar*) &dimg[0][0], cols, rows, QImage::Format_ARGB32);
  setPixmapForLabel(image, m_pixmap_cbar, m_lab_cbar);
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------

void 
WdgColorBar::message(QMouseEvent* e, const float& fraction, const char* comment) 
{
  std::stringstream ss; ss << comment 
                           << " : button " << int(e->button()) 
                           << " ratio from zero:" << std::fixed << std::setprecision(3) << fraction;
  MsgInLog(_name_(), DEBUG, ss.str());  
  //std::cout << ss.str() << '\n';
}

//--------------------------

void 
WdgColorBar::testPressColorBar(QMouseEvent* e, const float& ratio) 
{
  this->message(e, ratio, ":testPressColorBar");
}

//--------------------------

void 
WdgColorBar::testReleaseColorBar(QMouseEvent* e, const float& ratio) 
{
  this->message(e, ratio, "testReleaseColorBar");
}

//--------------------------

void 
WdgColorBar::testMoveOnColorBar(QMouseEvent* e, const float& ratio) 
{
  this->message(e, ratio, "testMoveOnColorBar");
}

//--------------------------

} // namespace PSQt

//--------------------------
