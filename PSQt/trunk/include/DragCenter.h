#ifndef PSQT_DRAGCENTER_H
#define PSQT_DRAGCENTER_H

//--------------------------
#include "PSQt/DragBase.h"

//#include <map>
#include <iostream>    // std::cout
#include <fstream>     // std::ifstream(fname)
#include <sstream>     // stringstream

namespace PSQt {

//--------------------------

// @addtogroup PSQt DragCenter

/**
 *  @ingroup PSQt DragCenter
 *
 *  @brief DragCenter - derived class for draggable circle.
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
 *  #include "PSQt/DragCenter.h"
 *  @endcode
 *
 *  @li Instatiation
 *  \n
 *  DragCenter inherits DragBase
 */

//--------------------------

class DragCenter : public DragBase
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:

  /**
   *  @brief DragCenter class for draggable center-mark
   *  
   *  @param[in] wimg - pointer to WdgImage
   *  @param[in] points - array of points which defines initial figure parameters
   */ 
    DragCenter(WdgImage* wimg=0, const QPointF* points=0); 
    virtual ~DragCenter(){}; 

    virtual void draw(const DRAGMODE& mode=DRAW);
    virtual bool contains(const QPointF& p);
    virtual void move(const QPointF& p);
    virtual void moveIsCompleted(const QPointF& p);
    virtual void create();

    virtual void print();

    virtual const QPointF& getCenter(){ return m_points_raw[0]; };
    //virtual void setCenter(const QPointF& p) { moveToRaw(p); };

    void forceToEmitSignal();

 signals:
    void centerIsMoved(const QPointF&);
    void centerIsChanged(const QPointF&);

 public slots:
    void moveToRaw(const QPointF&);
    void testSignalCenterIsChanged(const QPointF&);
    void testSignalCenterIsMoved(const QPointF&);

 protected:

 private:
   virtual  const char* _name_(){return "DragCenter";}

};

//--------------------------

} // namespace PSQt

#endif // PSQT_DRAGCENTER_H
//--------------------------
//--------------------------
//--------------------------

