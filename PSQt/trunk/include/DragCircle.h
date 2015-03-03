#ifndef PSQT_DRAGCIRCLE_H
#define PSQT_DRAGCIRCLE_H

//--------------------------
#include "PSQt/DragBase.h"

//#include <map>
#include <iostream>    // std::cout
#include <fstream>     // std::ifstream(fname)
#include <sstream>     // stringstream
 
namespace PSQt {

//--------------------------

// @addtogroup PSQt DragCircle

/**
 *  @ingroup PSQt DragCircle
 *
 *  @brief DragCircle - derived class for draggable circle.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see DragStore, WdgImageFigs, WdgImage
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
 *  #include "PSQt/DragCircle.h"
 *  @endcode
 *
 *  @li Instatiation
 *  \n
 *  DragCircle inherits DragBase
 */

//--------------------------

class DragCircle : public DragBase
{
 public:

  /**
   *  @brief DragCircle class for draggable circle
   *  
   *  @param[in] wimg - pointer to WdgImage
   *  @param[in] points - array of points which defines initial figure parameters
   */ 
    DragCircle(WdgImage* wimg=0, const QPointF* points=0); 
    virtual ~DragCircle(){}; 

    virtual void draw(const DRAGMODE& mode=DRAW);
    virtual bool contains(const QPointF& p);
    virtual void move(const QPointF& p);
    virtual void moveIsCompleted(const QPointF& p);
    virtual void create();

    virtual const QPointF& getCenter();
    virtual void print();

 protected:

 private:
    virtual const char* _name_(){return "DragCircle";}
    QPointF m_rad_raw;
};

//--------------------------

} // namespace PSQt

#endif // PSQT_DRAGCIRCLE_H
//--------------------------
//--------------------------
//--------------------------

