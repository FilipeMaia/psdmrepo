//--------------------------

#include "PSQt/QGUtils.h"

//#include "PSQt/WdgImage.h"
//#include "ndarray/ndarray.h" // for img_from_pixel_arrays(...)

#include <cstring>    // for memcpy
#include <iostream>   // for std::cout
#include <fstream>    // for ifstream
#include <sys/stat.h>

using namespace std;  // for cout without std::

namespace PSQt {


//--------------------------

uint32_t
fRGBA(const float R, const float G, const float B) // R, G, B = [0,1]
{
  unsigned r = unsigned(R*255);
  unsigned g = unsigned(G*255);
  unsigned b = unsigned(B*255);
  return uint32_t(0xFF000000 + (r<<16) + (g<<8) + b);  
}

//--------------------------

uint32_t 
HSV2RGBA(const float H, const float S, const float V) // H=[0,360), S, V = [0,1]
{
  // extend hue from [0,+360] to any angle in cycle
  float HE = fmod(H,360);

  float fhr = (HE<0) ? (HE+360)/60 : HE/60 ; 
  int   ihr = int(fhr); // [0,5]
  float rem = fhr - ihr;

  float p = V * (1 - S);
  float q = V * (1 - (S * rem));
  float t = V * (1 - (S * (1 - rem)));

  switch(ihr) {
    case  0: return fRGBA(V,t,p);
    case  1: return fRGBA(q,V,p);
    case  2: return fRGBA(p,V,t);
    case  3: return fRGBA(p,q,V);
    case  4: return fRGBA(t,p,V);
    case  5:
    default: return fRGBA(V,p,q);
  }
}

//--------------------------

uint32_t*
ColorTable(const unsigned& NColors, const float& H1, const float& H2) // H1, H2=[-360,360]
{
  uint32_t* ctable = new uint32_t[NColors];

  const float S=1; 
  const float V=1;
  const float dH = (H2-H1)/NColors;
  float H=H1;
  for(unsigned i=0; i<NColors; ++i, H+=dH) {
    ctable[i] = HSV2RGBA(H, S, V);
  }
  return ctable;
}

//--------------------------

ndarray<uint32_t,2>
getColorBarImage( const unsigned& rows, 
                  const unsigned& cols,
                  const float&    hue1,
                  const float&    hue2
                )
{
  MsgInLog("QGUtils", DEBUG, "getColorBarImage()");
  uint32_t* colors = ColorTable(cols, hue1, hue2);

  unsigned int shape[2] = {rows, cols};
  ndarray<uint32_t,2> img_nda(shape);

  for(unsigned r=0; r<rows; ++r) {
    std::memcpy(&img_nda[r][0], &colors[0], cols*sizeof(uint32_t));
  }

  delete[] colors;

  return img_nda;
}

//--------------------------

void 
setPixmapForLabel(const QImage& image, QPixmap*& pixmap, QLabel*& label)
{
  if(pixmap) delete pixmap;
  pixmap = new QPixmap(QPixmap::fromImage(image));
  //else pixmap -> loadFromData ( (const uchar*) &dimg[0], unsigned(rows*cols) );

  label->setPixmap(pixmap->scaled(label->size(), Qt::KeepAspectRatio, Qt::FastTransformation));
  //label->setPixmap(*pixmap);
}

//--------------------------

int string_to_int(const std::string& str)
{
  return atoi( str.c_str() );
}

//--------------------------

void print_font_status(const QFont& font)
{
  //QFont font(txtitem->font());
  std::cout << "Font family    : " << font.family().toStdString() << '\n';
  std::cout << "Font styleName : " << font.styleName().toStdString() << '\n';
  std::cout << "Font styleHint : " << font.styleHint() << '\n';
  std::cout << "Font style     : " << font.style() << '\n';
  std::cout << "Font weight    : " << font.weight() << '\n';
  std::cout << "Font stretch   : " << font.stretch() << '\n';
  std::cout << "Font bold      : " << font.bold() << '\n';
  std::cout << "Font kerning   : " << font.kerning() << '\n';
  std::cout << "Font rawMode   : " << font.rawMode() << '\n';
  std::cout << "Font rawName   : " << font.rawName().toStdString() << '\n';
  std::cout << "Font pointSize : " << font.pointSize() << '\n';
  std::cout << "Font pointSizeF: " << font.pointSizeF() << '\n';
  std::cout << "Font pixelSize : " << font.pixelSize() << '\n';
  std::cout << "Font strikeOut : " << font.strikeOut() << '\n';
}  

//--------------------------

void set_pen(QPen & pen, const std::string & opt)
{
  //QPen pen(Qt::blue,  2, Qt::SolidLine);
  //pen.setCosmetic(true);
  //pen.setStyle (Qt::SolidLine) // Qt::DashLine, Qt::DotLine, Qt::DashDotLine, Qt::DashDotDotLine, Qt::NoPen	

    if( opt.find('-') != std::string::npos ) pen.setStyle (Qt::SolidLine);
    if( opt.find('.') != std::string::npos ) pen.setStyle (Qt::DotLine);
    if( opt.find('!') != std::string::npos ) pen.setStyle (Qt::DashDotLine);

    if( opt.find('k') != std::string::npos ) pen.setColor(Qt::black);
    if( opt.find('b') != std::string::npos ) pen.setColor(Qt::blue);
    if( opt.find('r') != std::string::npos ) pen.setColor(Qt::red);
    if( opt.find('g') != std::string::npos ) pen.setColor(Qt::green);
    if( opt.find('y') != std::string::npos ) pen.setColor(Qt::yellow);

    if( opt.find('1') != std::string::npos ) pen.setWidth(1);
    if( opt.find('2') != std::string::npos ) pen.setWidth(2);
    if( opt.find('3') != std::string::npos ) pen.setWidth(3);
    if( opt.find('4') != std::string::npos ) pen.setWidth(4);
    if( opt.find('5') != std::string::npos ) pen.setWidth(5);
}
//--------------------------

void set_brash(QBrush & brush, const std::string & opt)
{
  //QBrush brush(Qt::transparent, Qt::SolidPattern);  
  //brush.setStyle (Qt::SolidPattern);

    if( opt.find('T') != std::string::npos ) brush.setColor(Qt::transparent);
    if( opt.find('K') != std::string::npos ) brush.setColor(Qt::black);
    if( opt.find('B') != std::string::npos ) brush.setColor(Qt::blue);
    if( opt.find('R') != std::string::npos ) brush.setColor(Qt::red);
    if( opt.find('G') != std::string::npos ) brush.setColor(Qt::green);
    if( opt.find('Y') != std::string::npos ) brush.setColor(Qt::yellow);
}

//--------------------------

void set_pen_brush(QPen & pen, QBrush & brush, const std::string & opt)
{
  if(opt.empty()) return;
  set_pen(pen, opt);
  set_brash(brush, opt);
}

//--------------------------

void graph(QGraphicsScene* scene, const float* x, const float* y, const int n, const std::string& opt)
{
  //std::cout << "QGraphicsScene\n";
  //QPen pen;
  QPen   pen  (Qt::blue, 2, Qt::SolidLine);
  QBrush brush(Qt::transparent, Qt::SolidPattern);  

  set_pen_brush(pen, brush, opt);

  pen.setCosmetic(true);

  QPainterPath path;     path.moveTo(x[0],y[0]);
  for(int i=1; i<n; ++i) path.lineTo(x[i],y[i]);
  scene->addPath(path, pen, brush);
}

//--------------------------

void graph(QGraphicsView * view, const float* x, const float* y, const int n, const std::string& opt)
{
  //std::cout << "QGraphicsView\n";
  graph(view->scene(), x, y, n, opt);
}

//--------------------------

void graph(PSQt::GUAxes * axes, const float* x, const float* y, const int n, const std::string& opt)
{
  //std::cout << "PSQt::GUAxes\n";
  graph(axes->pview()->scene(), x, y, n, opt);
}

//--------------------------

bool is_directory(const std::string& path)
{
  struct stat st;
  if(stat(path.c_str(),&st) == 0) 
    if ((st.st_mode & S_IFMT) == S_IFDIR) return true;
  return false;
}

//--------------------------

bool is_link(const std::string& path)
{
  struct stat st;
  if(stat(path.c_str(),&st) == 0) 
    if ((st.st_mode & S_IFMT) == S_IFLNK) return true;
  return false;
}

//--------------------------

bool is_file(const std::string& path)
{
  struct stat st;
  if(stat(path.c_str(),&st) == 0) 
    if ((st.st_mode & S_IFMT) == S_IFREG) return true;
  return false;
}

//--------------------------

bool dir_exists(const std::string& dir)
{    
  return is_directory(dir);
}

//--------------------------

bool path_exists(const std::string& path) {
  struct stat st;   
  return (stat(path.c_str(), &st) == 0); 
}


//--------------------------


bool file_exists(const std::string& fname)
{
    std::ifstream f(fname.c_str());
    return (f.good())? true : false;
}


//--------------------------

std::string
split_string_left(const std::string& s, size_t& pos, const char& sep)
{
  size_t p0 = pos;
  size_t p1 = s.find(sep, p0);
  size_t nchars = p1-p0; 
  pos = p1+1; // move position to the next character after separator

  if (p1 != std::string::npos) return std::string(s,p0,nchars);
  else if (p0 < s.size())      return std::string(s,p0);
  else                         return std::string();
}

//--------------------------
// for input path = <path>/<fname> returns <fname>
std::string basename(const std::string& path)
{
  if (path.empty()) return path;
  size_t pos = path.find_last_of('/');
  if (pos == std::string::npos) return path;
  else if (pos < path.size()-1) return std::string(path,pos+1);
  else return std::string();
}

//--------------------------
// for input path = <path>/<fname> returns <path>
std::string dirname(const std::string& path)
{
  if (path.empty()) return path;
  if (path==std::string(".")) return path;
  if (path==std::string("..")) return path;

  size_t pos = path.find_last_of('/');
  if (pos == std::string::npos) return std::string(); // assume that path is a file name
  else return std::string(path,0,pos);
}

//--------------------------
// for input path = "<path>/<fname>.<ext>" returns root="<path>/<fname>" and ext=".<ext>"
void splitext(const std::string& path, std::string& root, std::string& ext)
{
  if (path.empty()) {
    root = std::string();
    ext  = std::string();
    return;
  }

  if(path==std::string(".")) {
    root = path;
    ext  = std::string();
    return;
  }

  size_t pos = path.find_last_of('.');

  if (pos == std::string::npos) {
    root = path;
    ext  = std::string();
    return;
  }
  else {
    root = std::string(path,0,pos);
    ext  = std::string(path,pos);
    return;
  }
}

//--------------------------

std::string stringTimeStamp(const std::string& format)
{
  time_t  time_sec;
  time ( &time_sec );
  struct tm* timeinfo; timeinfo = localtime ( &time_sec );
  char c_time_buf[32]; strftime(c_time_buf, 32, format.c_str(), timeinfo);
  return std::string (c_time_buf);
}

//--------------------------

std::string
getFileNameWithTStamp(const std::string& fname)
{
  std::string ofname = (fname==std::string()) ? "file.txt" : fname;
  std::string root, ext;
  splitext(ofname, root, ext);
  return root + '-' + strTimeStamp() + ext; 
}

//--------------------------

std::string
getGeometryFileName(const std::string& fname, const bool& add_tstamp)
{
  return (add_tstamp) ? getFileNameWithTStamp(fname) : fname;  
}

//--------------------------

} // namespace PSQt

//--------------------------
