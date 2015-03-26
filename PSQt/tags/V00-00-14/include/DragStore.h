#ifndef PSQT_DRAGSTORE_H
#define PSQT_DRAGSTORE_H

//--------------------------

#include "PSQt/DragBase.h"

//--------------------------

namespace PSQt {

//--------------------------
/**
 *  @ingroup PSQt DRAGTYPE
 *
 *  @brief DRAGTYPE - enumerator for DragBase type objects
 */ 

  enum DRAGTYPE {DRAGCIRCLE=0, DRAGCENTER, DRAGLINE, DRAGNONE};

//--------------------------
/**
 *  @ingroup PSQt Record
 *
 *  @brief DragFig - struct for DragStore
 */ 

struct DragFig {
  DragBase* ptr_obj;
  DRAGTYPE  type; 
  unsigned  flags;

  //std::string strRecordTotal();
  //std::string strRecordBrief();
  //std::string strRecord();
};


/// @addtogroup PSQt PSQt

/**
 *  @ingroup PSQt
 *
 *  @brief Contains static factory method Create for DragBase objects.
 *
 *  This software was developed for the LCLS project. If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @version $Id$
 *
 *  @author Mikhail S. Dubrovin
 *
 *  @see Drag, DragCspad2x1V1
 *
 *  @anchor interface
 *  @par<interface> Interface Description
 * 
 *  @li  Includes
 *  @code
 *  // #include "PSQt/DragBase.h" // already included under DragStore.h
 *  #include "PSQt/DragStore.h"
 *  typedef PSQt::DragBase DF;
 *  @endcode
 *
 *  @li Instatiation
 *  \n
 *  Classes like DragCircle containing implementation of the DragBase interface methods are self-sufficient. 
 *  Factory method Create should returns the pointer to the DragBase object for specified figure parameter or returns 0-pointer if segname is not recognized (and not implemented).
 *  Code below instateates DragBase object using factory static method PSQt::DragStore::Create()
 *  @code
 *  DRAGTYPE dragfig = DRAGCIRCLE; // or DRAGCENTER, DRAGLINE, etc.;
 *  DF* dfigobj = PSQt::DragStore::Create(dragfig);
 *  @endcode
 *
 *  @li Print info
 *  @code
 *  dfigobj -> print(0377);
 *  @endcode
 *
 *  @li Access methods
 *  \n are defined in the interface DragBase and implemented in DragCircle etc.
 *  @code
 *  // scalar values
 *  const DF::size_t array_size = dfigobj -> size(); 
 *  @endcode
 *
 *  @li How to add new segment to the factory
 *  \n 1. implement DragBase interface methods in class like DragCircle
 *  \n 2. add it to DragStore with unique dragfig
 */

//--------------------------

class DragStore {

 public:

  DragStore (WdgImage* wimg = 0);
  virtual ~DragStore () {}

  void print();
  void drawFigs(const DRAGMODE& mode=DRAW);
  bool containFigs(const QPointF& p);
  void moveFigs(const QPointF& p);
  void moveFigsIsCompleted(const QPointF& p);
  void addCircle(const float& rad_raw=100, const QPen* pen=0); // QPen(Qt::white, 1, Qt::SolidLine));
  void deleteFig();

  /**
   *  @brief Static factory method to create object of classes derived from DragBase
   *  
   *  @param[in] wimg - pointer to WdgImage
   *  @param[in] dfigtype - figure enumerated type
   *  @param[in] points - array of points which defines initial figure parameters
   *  @param[in] npoints - number of points in the array
   */
  static PSQt::DragBase* Create(WdgImage* wimg, const DRAGTYPE& dfigtype=DRAGCIRCLE, const QPointF* points=0, const int& npoints=0);
  static const char* cstrDragType(const DRAGTYPE& dfigtype=DRAGCIRCLE);
  static inline const char* _name_(){return "DragStore";} 

  const QPointF& getCenter(){ return v_dragfigs[0].ptr_obj->getCenter(); }
  const DragBase* getDragCenter(){ return v_dragfigs[0].ptr_obj; }


 private:

  WdgImage*            m_wimg;
  std::vector<DragFig> v_dragfigs;
  DragFig*             p_dragfig_sel;
};

} // namespace PSQt

#endif // PSQT_DRAGSTORE_H

//--------------------------
