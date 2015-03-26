#ifndef PSQT_DRAGBASE_H
#define PSQT_DRAGBASE_H

//--------------------------
#include "PSQt/QGUtils.h"
#include "PSQt/Logger.h"
#include "PSQt/WdgImage.h"

#include <QtCore>

//#include <map>
#include <iostream>    // std::cout
#include <fstream>     // std::ifstream(fname)
#include <sstream>     // stringstream
#include <iomanip>     // setw, setfill

using namespace std;   // cout without std::
 
namespace PSQt {

//--------------------------
/**
 *  @ingroup PSQt DRAGMODE
 *
 *  @brief DRAGMODE - enumerator of modes for DragBase
 */ 

  enum DRAGMODE {ZOOM=0, CREATE, DELETE, MOVE, DRAW};

//--------------------------

// @addtogroup PSQt DragBase

/**
 *  @ingroup PSQt DragBase
 *
 *  @brief Base class for draggable figures like circle, line, center, etc on the plot.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see DragStore, DragCenter, DragCircle, WdgImageFigs, WdgImage
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 *
 *
 *
 *
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Include
 *  @code
 *  #include "PSQt/DragBase.h"
 *  @endcode
 *
 *  @li Instatiation
 *  \n
 *  DragBase is inhereted and used by the class
 */

//--------------------------

class DragBase : public QObject
{
  // Q_OBJECT // macro is needed for connection of signals and slots

 public:

  /**
   *  @brief DragBase - base class for draggable figures
   *  
   *  @param[in] wimg - pointer to WdgImage
   *  @param[in] points - array of points which defines initial figure parameters
   *  @param[in] npoints - number of points in the array
   */ 

    DragBase(WdgImage* wimg, const QPointF* points, const int& npoints=1); 
    virtual ~DragBase(); 

    virtual void draw(const DRAGMODE& mode=DRAW) = 0;
    virtual bool contains(const QPointF& p) = 0;
    virtual void move(const QPointF& p) = 0;
    virtual void moveIsCompleted(const QPointF& p) = 0;
    virtual void create() = 0;

    virtual const QPointF& getCenter() { return m_center_def; };
    virtual void print();

    void setPenMove(const QPen& pen) { m_pen_move = pen; };
    void setPenDraw(const QPen& pen) { m_pen_draw = pen; };
    void setPickRadius(const float& rpick) { m_rpick = rpick; };

 protected:

    void setImagePointsFromRaw();
    void setRawPointsFromImage();

    PSQt::WdgImage* m_wimg;
    QPointF*        m_points_raw;
    QPointF*        m_points_img;
    int             m_npoints;
    QPen            m_pen_draw;
    QPen            m_pen_move;
    float           m_rpick;

 private:
    virtual  const char* _name_(){return "DragBase";}
    void copyRawPoints(const QPointF* points=0);
    QPointF         m_center_def; 

 };

//--------------------------

} // namespace PSQt

#endif // PSQT_DRAGBASE_H
//--------------------------
//--------------------------
//--------------------------

