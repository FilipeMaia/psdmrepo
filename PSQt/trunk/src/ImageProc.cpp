//--------------------------
#include "PSQt/ImageProc.h"
#include "PSQt/Logger.h"
#include "PSQt/QGUtils.h"

#include <iostream>    // for std::cout
#include <fstream>     // for std::ifstream(fname)
#include <cstdlib>     // for rand()
#include <cstring>     // for memcpy

#include <math.h>      // atan2
#include <algorithm> // for fill_n
//using namespace std; // for cout without std::

namespace PSQt {

//--------------------------

ImageProc::ImageProc()
  : QObject(NULL)
  , m_rbin_width(1)
  , m_image_is_set(false)
  , m_center_is_set(false)
  , m_zoom_is_set(false)
  , m_rindx_is_set(false)
  , m_rhist_is_set(false)
  , m_ixmax(0)
  , m_iymax(0)
  , m_irmax(0)
  , m_zxmin(0)
  , m_zymin(0)
  , m_zxmax(0)
  , m_zymax(0)
  , p_rsum(0)
  , p_rsta(0)
{
  connect(this, SIGNAL(rhistIsFilled(ndarray<float, 1>&)), 
          this, SLOT(testSignalRHistIsFilled(ndarray<float, 1>&)));
}

//--------------------------

ImageProc::~ImageProc()
{
  delete [] p_rsum;
  delete [] p_rsta;
}

//--------------------------
/*
void
ImageProc::connectSignalsAndSlots()
{
  connect(xxxx, SIGNAL(imageIsUpdated(const ndarray<const GeoImage::raw_image_t,2>&)), 
          this, SLOT(onImageIsUpdated(const ndarray<const GeoImage::raw_image_t,2>&)));

  connect(xxxx, SIGNAL(centerIsChanged(const QPointF&)), 
          this, SLOT(onCenterIsChanged(const QPointF&)));

  connect(xxxx, SIGNAL(zoomIsChanged(int&, int&, int&, int&)), 
          this, SLOT(testSignalZoomIsChanged(int&, int&, int&, int&)));
}
*/

//--------------------------
//--------------------------
//--------- SLOTS ----------
//--------------------------
//--------------------------

void 
ImageProc::onImageIsUpdated(const ndarray<const GeoImage::raw_image_t,2>& nda)
{ 
  m_nda_image = nda.copy();
  m_image_is_set = true;

  stringstream ss; ss << "onImageIsUpdated(), size = " << nda.size() 
                      << " shape = (" << nda.shape()[0] << ", " << nda.shape()[1] << ")";
  MsgInLog(_name_(), INFO, ss.str()); 

  // if the image shape is changed, then indexes need to be re-evaluated
  if(nda.shape()[0] != m_iymax
  || nda.shape()[1] != m_ixmax) this->evaluateRadialIndexes();
}

//--------------------------
void 
ImageProc::onCenterIsChanged(const QPointF& pc)
{
  m_center = pc;
  m_center_is_set = true; 

  stringstream ss; ss << "::onCenterIsChanged() x: " << fixed << std::setprecision(1) << pc.x() << "  y: " << pc.y();  
  MsgInLog(_name_(), INFO, ss.str());
  //std::cout << ss.str() << '\n';

  this->evaluateRadialIndexes();
}

//--------------------------
void 
ImageProc::onZoomIsChanged(int& xmin, int& ymin, int& xmax, int& ymax)
{
  m_zoom_is_set = true;

  m_zxmin = xmin; 
  m_zymin = ymin;
  m_zxmax = xmax;
  m_zymax = ymax;

  stringstream ss;
  ss << "onZoomIsChanged(...): zoom is changed to"
     << "  xmin=" << m_zxmin 
     << "  ymin=" << m_zymin
     << "  xmax=" << m_zxmax 
     << "  ymax=" << m_zymax;
  MsgInLog(_name_(), INFO, ss.str());

  this->fillRadialHistogram();
  this->fillSpectralHistogram();
}

//--------------------------
bool 
ImageProc::evaluateRadialIndexes()
{
  m_rindx_is_set = false;

  if(! m_image_is_set) {
    MsgInLog(_name_(), WARNING, "::evaluateRadialIndexes() - image is not set, endexes are not evaluated");
    return false;
  }

  if(! m_center_is_set) {
    MsgInLog(_name_(), WARNING, "::evaluateRadialIndexes() - center is not set, endexes are not evaluated");
    return false;
  }

  m_iymax = m_nda_image.shape()[0];
  m_ixmax = m_nda_image.shape()[1];
  m_nda_rindx = make_ndarray<unsigned>(m_iymax, m_ixmax);

  float dx, dy, dr;
  float bin_scf = 1./m_rbin_width;

  unsigned ir;
  m_irmax = 0;

  for(unsigned iy=0; iy<m_iymax; ++iy) {
    dy = float(iy) - m_center.y();

    for(unsigned ix=0; ix<m_ixmax; ++ix) {
      dx = float(ix) - m_center.x();

      dr = sqrt(dx*dx + dy*dy);
      ir = unsigned(dr*bin_scf);
      if(ir>m_irmax) m_irmax = ir;
      m_nda_rindx[iy][ix] = unsigned(ir);
    }
  }

  m_irmax+=1;

  stringstream ss; ss <<  "::evaluateRadialIndexes() - r-indexes are evaluated. irmax = " << m_irmax;
  MsgInLog(_name_(), INFO, ss.str()); 

  m_rindx_is_set = true;
  return true;
}

//--------------------------
bool 
ImageProc::fillRadialHistogram()
{
  m_rhist_is_set = false;

  if(! m_rindx_is_set) 
    if(! this->evaluateRadialIndexes()) {
      MsgInLog(_name_(), WARNING, "::onZoomIsChanged() - radial indexes are not available");
      return false;
    }

  if(p_rsum) delete [] p_rsum;
  if(p_rsta) delete [] p_rsta;

  p_rsum = new float[m_irmax];
  p_rsta = new unsigned[m_irmax];

  std::fill_n(p_rsum, int(m_irmax), float(0));
  std::fill_n(p_rsta, int(m_irmax), unsigned(0));

  unsigned ir;

  for(int iy=m_zymin; iy<m_zymax; ++iy) {
    for(int ix=m_zxmin; ix<m_zxmax; ++ix) {
      ir = m_nda_rindx[iy][ix];
      p_rsum[ir] += m_nda_image[iy][ix];
      p_rsta[ir] ++;
    }
  }

  m_nda_rhist = make_ndarray<float>(m_irmax);
  for(unsigned i=0; i<m_irmax; ++i) { m_nda_rhist[i] = (p_rsta[i]) ? p_rsum[i]/p_rsta[i] : 0; }

  //for(unsigned i=0; i<200; ++i) std::cout << i << ":  " << p_rsta[i] << ":  " << m_nda_rhist[i] << '\n';

  //ndarray<unsigned, 2>::iterator it;
  //for(it=m_nda_rindx.begin(); it!=m_nda_rindx.end(); it++) { p_data[ind++] = double(int(*cit) - offset); }

  emit rhistIsFilled(m_nda_rhist);

  m_rhist_is_set = true;
  return true;
}

//--------------------------
bool 
ImageProc::fillSpectralHistogram()
{
  return true;
}

//--------------------------
void 
ImageProc::testSignalRHistIsFilled(ndarray<float, 1>& nda)
{
  stringstream ss; ss <<  "::testSignalRHistIsFilled() size()=" << nda.size();
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


