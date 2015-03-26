//---------------------------------------------------------------------
// File and Version Information:
//   $Id$
//
// Author: Mikhail S. Dubrovin
//---------------------------------------------------------------------

//--------------------------
#include "PSQt/ImageProc.h"
#include "PSQt/Logger.h"
#include "PSQt/QGUtils.h"

#include <iostream>    // for std::cout
#include <fstream>     // for std::ifstream(fname)
#include <cstdlib>     // for rand()
#include <cstring>     // for memcpy

#include <math.h>      // atan2, floor
#include <algorithm> // for max, fill_n
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
  , m_shist_is_set(false)
  , m_ixmax(0)
  , m_iymax(0)
  , m_irmax(0)
  , m_zxmin(0)
  , m_zymin(0)
  , m_zxmax(0)
  , m_zymax(0)
  , m_amin(0)
  , m_amax(0)
		    //  , p_ssta(0)
		    //  , p_rsum(0)
		    //  , p_rsta(0)
{
  connect(this, SIGNAL(rhistIsFilled(ndarray<float, 1>&, const unsigned&, const unsigned&)), 
          this, SLOT(testSignalRHistIsFilled(ndarray<float, 1>&, const unsigned&, const unsigned&)));

  connect(this, SIGNAL(shistIsFilled(float*, const float&, const float&, const unsigned&)), 
          this, SLOT(testSignalSHistIsFilled(float*, const float&, const float&, const unsigned&)));
}

//--------------------------

ImageProc::~ImageProc()
{
  if(p_rsum) delete [] p_rsum;
  if(p_rsta) delete [] p_rsta;
  if(p_ssta) delete [] p_ssta;
}

//--------------------------
/*
void
ImageProc::connectSignalsAndSlots()
{
  connect(xxxx, SIGNAL(imageIsUpdated(ndarray<GeoImage::raw_image_t,2>&)), 
          this, SLOT(onImageIsUpdated(ndarray<GeoImage::raw_image_t,2>&)));

  connect(xxxx, SIGNAL(centerIsChanged(const QPointF&)), 
          this, SLOT(onCenterIsChanged(const QPointF&)));

  connect(xxxx, SIGNAL(zoomIsChanged(int&, int&, int&, int&, float&, float&)), 
          this, SLOT(onZoomIsChanged(int&, int&, int&, int&, float&, float&)));
}
*/

//--------------------------
//--------------------------
//--------- SLOTS ----------
//--------------------------
//--------------------------

void 
ImageProc::onImageIsUpdated(ndarray<GeoImage::raw_image_t,2>& nda)
{ 
  //m_nda_image = nda.copy();
  m_nda_image = nda;
  m_image_is_set = true;
  m_rhist_is_set = false;
  m_zoom_is_set = false;

  stringstream ss; ss << "::onImageIsUpdated(), size = " << nda.size() 
                      << " shape = (" << nda.shape()[0] << ", " << nda.shape()[1] << ")";
  MsgInLog(_name_(), DEBUG, ss.str()); 

  m_zxmin = 0; 
  m_zymin = 0;
  m_zxmax = nda.shape()[1] - 1;
  m_zymax = nda.shape()[0] - 1;

  this->fillSpectralHistogram(m_amin,m_amax);

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

  stringstream ss; ss << "Set center in position x=" << fixed << std::setprecision(1) << pc.x() << ", y=" << pc.y();  
  MsgInLog(_name_(), DEBUG, ss.str());
  //std::cout << ss.str() << '\n';

  this->evaluateRadialIndexes();
  this->fillRadialHistogram();
}

//--------------------------
void 
ImageProc::onZoomIsChanged(int& xmin, int& ymin, int& xmax, int& ymax, float& amin, float& amax)
{
  if      (xmin==0 && xmax==0 && ymin==0 && ymax==0) m_zoom_is_set = false;
  else if (xmin>0) m_zoom_is_set = true;
  else if (ymin>0) m_zoom_is_set = true; 
  else if ((unsigned)xmax<m_ixmax-1) m_zoom_is_set = true; 
  else if ((unsigned)ymax<m_iymax-1) m_zoom_is_set = true; 
  else m_zoom_is_set = false;

  m_zxmin = xmin; 
  m_zymin = ymin;
  m_zxmax = xmax;
  m_zymax = ymax;

  if(!(xmax<(int)m_nda_image.shape()[1])) {
    m_zxmax = m_nda_image.shape()[1];
    m_rindx_is_set = false;
  }
  if(!(ymax<(int)m_nda_image.shape()[0])) {
    m_zymax = m_nda_image.shape()[0];
    m_rindx_is_set = false;
  }

  stringstream ss;
  ss << "Zoom is set to"
     << " xmin=" << m_zxmin 
     << " ymin=" << m_zymin
     << " xmax=" << m_zxmax 
     << " ymax=" << m_zymax
     << " amin=" << amin 
     << " amax=" << amax
     << " m_zoom_is_set=" << m_zoom_is_set;

  MsgInLog(_name_(), INFO, ss.str());

  this->fillRadialHistogram();
  this->fillSpectralHistogram(amin, amax);
}

//--------------------------
void 
ImageProc::evaluateRadialIndexes()
{
  m_rindx_is_set = false;

  if(! m_image_is_set) {
    MsgInLog(_name_(), DEBUG, "::evaluateRadialIndexes() - image is not set, indexes are not evaluated");
    return; // false;
  }

  if(! m_center_is_set) {
    MsgInLog(_name_(), DEBUG, "::evaluateRadialIndexes() - center is not set, indexes are not evaluated");
    return; // false;
  }

  m_iymax = m_nda_image.shape()[0];
  m_ixmax = m_nda_image.shape()[1];

  stringstream ss;
  ss << "::evaluateRadialIndexes:"
     << "  m_ixmax:" << m_ixmax 
     << "  m_iymax:" << m_iymax;
  MsgInLog(_name_(), DEBUG, ss.str()); 

  m_nda_rindx = make_ndarray<unsigned>(m_iymax, m_ixmax);
  //if(m_nda_image.empty()) 

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

  stringstream ss2; ss2 <<  "::evaluateRadialIndexes() - r-indexes are evaluated. irmax = " << m_irmax;
  MsgInLog(_name_(), DEBUG, ss2.str()); 

  m_rindx_is_set = true;
  return; // true;
}

//--------------------------
void
ImageProc::fillRadialHistogram()
{
  MsgInLog(_name_(), DEBUG, "::fillRadialHistogram()");

  m_rhist_is_set = false;

  //this->evaluateRadialIndexes();

  if(! m_rindx_is_set) {
    this->evaluateRadialIndexes();
    if(! m_rindx_is_set) {
      MsgInLog(_name_(), DEBUG, "::fillRadialHistogram() - radial indexes are not available");
      return; // false;
    }
  }

  if (m_zoom_is_set) {
    m_zirmax = std::max(
                 std::max(m_nda_rindx[m_zymin][m_zxmin],
                          m_nda_rindx[m_zymin][m_zxmax]),
                 std::max(m_nda_rindx[m_zymax][m_zxmin],
                          m_nda_rindx[m_zymax][m_zxmax])
	       );
  }
  else m_zirmax = m_irmax;

  stringstream ss1; 
  ss1 << "  fillRadialHistogram:"
      << "  shape()[1]:" << m_nda_image.shape()[1] 
      << "  shape()[0]:" << m_nda_image.shape()[0]
      << "\nm_zxmin:" << m_zxmin 
      << "  m_zxmax:" << m_zxmax 
      << "  m_zymin:" << m_zymin 
      << "  m_zymax:" << m_zymax 
      << "\n  number of bins in radial histogram = " << m_zirmax;  
  MsgInLog(_name_(), DEBUG, ss1.str());

  //if(p_rsum) delete [] p_rsum;
  //if(p_rsta) delete [] p_rsta;
  //p_rsum = new float[m_zirmax];
  //p_rsta = new unsigned[m_zirmax];

  m_zirmax = (m_zirmax<2000) ? m_zirmax : 2000-1;

  std::fill_n(p_rsum, int(m_zirmax), float(0));
  std::fill_n(p_rsta, int(m_zirmax), unsigned(0));

  std::stringstream ss; 
  ss << "CHECK POINT 1"
     << "  m_zxmin:" << m_zxmin
     << "  m_zxmax:" << m_zxmax
     << "  m_zymin:" << m_zymin
     << "  m_zymax:" << m_zymax
     << "\n    m_nda_image.shape:" << m_nda_image.shape()[0] << ", " << m_nda_image.shape()[1]
     << "\n    m_nda_rindx.shape:" << m_nda_rindx.shape()[0] << ", " << m_nda_rindx.shape()[1];
  MsgInLog(_name_(), DEBUG, ss.str());

  m_zirmin = m_zirmax;
  unsigned ir=0;

  for(int iy=m_zymin; iy<m_zymax; ++iy) {
    for(int ix=m_zxmin; ix<m_zxmax; ++ix) {
      ir = m_nda_rindx[iy][ix];
      if(ir<m_zirmin) m_zirmin = ir; 
      if(ir>m_zirmax) ir=m_zirmax-1; 
      p_rsum[ir] += m_nda_image[iy][ix];
      p_rsta[ir] ++;
    }
  }

  m_nda_rhist = make_ndarray<float>(m_zirmax);
  for(unsigned i=0; i<m_zirmax; ++i) { m_nda_rhist[i] = (p_rsta[i]) ? p_rsum[i]/p_rsta[i] : 0; }

  MsgInLog(_name_(), DEBUG, "CHECK  POINT 2: statistics for radial histogram is accumulated successfully.");

  MsgInLog(_name_(), DEBUG, "emit rhistIsFilled(...)");

  emit rhistIsFilled(m_nda_rhist, m_zirmin, m_zirmax);

  m_rhist_is_set = true;
  return; // true;
}

//--------------------------
void 
ImageProc::getIntensityLimits(float& imin, float& imax)
{
    float a(0);
    imin = (float)m_nda_image[m_zymin][m_zxmin];
    imax = imin;
    for(int iy=m_zymin; iy<m_zymax; ++iy) {
      for(int ix=m_zxmin; ix<m_zxmax; ++ix) {
        a = (float)m_nda_image[iy][ix];
        if(a<imin) imin = a;
        if(a>imax) imax = a;
      }
    }
}

//--------------------------
void 
ImageProc::fillSpectralHistogram(const float& amin, const float& amax, const unsigned& nbins)
{
  MsgInLog(_name_(), DEBUG, "::fillSpectralHistogram()");
  //std::cout << "::fillSpectralHistogram()"; 

  m_shist_is_set = false;

  if(!m_image_is_set) { 
    MsgInLog(_name_(), DEBUG, "::fillSpectralHistogram() - image is not available - can't fill spectrum...");    
    return; // false;
  }

  std::stringstream ss; ss << "Input range amin, amax: " << amin << ", "  << amax;
  MsgInLog(_name_(), DEBUG, ss.str());

  //if(!m_zoom_is_set) return; // false;

  m_nbins = nbins;
  float a(0);

  //if(p_ssta) delete [] p_ssta;
  //p_ssta = new float[m_nbins];

  m_nbins = (m_nbins<1000) ? m_nbins : 1000-1;
  
  std::fill_n(p_ssta, int(m_nbins), float(0));

  if(amin && amax) {
    m_amin = amin;
    m_amax = amax;
  }
  else {
    float Imin, Imax;
    //getIntensityLimits(Imin,Imax);
    getMinMax<GeoImage::raw_image_t>(m_nda_image, Imin, Imax);
    m_amax = (amax==0) ? Imax : amax;
    m_amin = Imin;
  }

  std::stringstream ss2; ss2 << "Evaluate histogram for range: " << m_amin << ", "  << m_amax;
  MsgInLog(_name_(), DEBUG, ss2.str());

  unsigned ih(0);
  float binw = (m_amax-m_amin)/m_nbins;

  for(int iy=m_zymin; iy<m_zymax; ++iy) {
    for(int ix=m_zxmin; ix<m_zxmax; ++ix) {

      a = (float)m_nda_image[iy][ix];
      if     (a<m_amin) continue; // ih = 0;
      else if(a<m_amax) ih = unsigned((a-m_amin)/binw);
      else              continue; // ih = m_nbins-1;
      p_ssta[ih] ++;
    }
  }

  //m_nda_shist = make_ndarray<float>(m_zirmax);
  emit shistIsFilled(p_ssta, m_amin, m_amax, m_nbins);

  m_shist_is_set = true;
  return; // true;
}

//--------------------------

/*
void 
ImageProc::onPressOnAxes(QMouseEvent* e, QPointF p)
{
  std::stringstream ss;
  ss << _name_() << " onPressOnAxes"
     << "  button: " << e->button()
     << "  window x(), y() = " << e->x() << ", " << e->y()
     << "  scene x(), y() = " << p.x() << ", " << p.y();

  MsgInLog(_name_(), INFO, ss.str());

  float amin = m_amin;
  float amax = m_amax;

  switch (e->button()) {
  case Qt::LeftButton  : amin = p.x(); break;
  case Qt::RightButton : amax = p.x(); break;
  default : 
  case Qt::MidButton   : amin = 0; amax = 0; break;
  }

  fillSpectralHistogram(amin, amax);
}
*/

//--------------------------
void 
ImageProc::testSignalRHistIsFilled(ndarray<float, 1>& nda, const unsigned& zirmin, const unsigned& zirmax)
{
  stringstream ss; ss <<  "::testSignalRHistIsFilled() size()=" << nda.size()
                      <<  "  zirmin=" << zirmin
                      <<  "  zirmax=" << zirmax;
  MsgInLog(_name_(), DEBUG, ss.str());   
}

//--------------------------
void 
ImageProc::testSignalSHistIsFilled(float* p, const float& amin, const float& amax, const unsigned& nbins)
{
  stringstream ss; ss <<  "::testSignalSHistIsFilled(): nbins=" << nbins
                      <<  "  amin=" << amin
                      <<  "  amax=" << amax;
  MsgInLog(_name_(), DEBUG, ss.str());
}

//--------------------------
//--------------------------
//--------------------------
//--------------------------
//--------------------------

} // namespace PSQt

//--------------------------


