#ifndef PSQT_WDGRADHIST_H
#define PSQT_WDGRADHIST_H

#include "ndarray/ndarray.h"
#include <stdint.h> // uint8_t, uint32_t, etc.

#include <PSQt/GUAxes.h>

#include <QWidget>
//#include <QLabel>
//#include <QFrame>
//#include <QPainter>
//#include <QPen>

//#include <QPoint>
//#include <QRect>

//#include <QPixmap>
//#include <QImage>
//#include <QBitmap>

//#include <QCloseEvent>
//#include <QResizeEvent>
//#include <QMouseEvent>

//#include <QScrollBar>
//#include <QScrollArea>

//#include <Qt>
#include <QtGui>
#include <QtCore>


namespace PSQt {

/**
 *  @ingroup PSQt
 * 
 *  @brief Widget to display angular-integrated radial-projection historgam for (zoomed) image.
 * 
 *  @code
 *  public slots:
 *    void onRHistIsFilled(ndarray<float, 1>&, const unsigned&, const unsigned&);
 *  @endcode
 *
 *  Constructor creates a widget with default frame of axes.
 *  Actual axes scale and histogram show up through the call to slot onRHistIsFilled(...).
 *  It is assumed that pixels have square geometry and radial bin size is equal to the pixel side size.
 *
 *  This software was developed for the LCLS project.  If you use all or 
 *  part of it, please give an appropriate acknowledgment.
 *
 *  @see GUView
 *
 *  @version $Id$
 *
 *  @author Mikhail Dubrovin
 */

class WdgRadHist : public QWidget
{
 Q_OBJECT // macro is needed for connection of signals and slots

 public:
  WdgRadHist(QWidget *parent = 0); 

  virtual ~WdgRadHist();

 public slots:
  /**
   *  @brief Public slot for histogram update 
   *  
   *  @param[in] - ndarray<float, 1>& - ndarray with histogram data from zero-radius to maximal for zoomed image
   *  @param[in] - const unsigned& - low index in the histogram array for zoomed region
   *  @param[in] - const unsigned& - upper index in the histogram array for zoomed region
   */ 
    void onRHistIsFilled(ndarray<float, 1>&, const unsigned&, const unsigned&);

 private:
    inline const char* _name_(){return "WdgRadHist";}

    QVBoxLayout* m_vbox;
    PSQt::GUAxes* m_axes;
    QGraphicsPathItem* m_path_item;
};

} // namespace PSQt

#endif // PSQT_WDGRADHIST_H
