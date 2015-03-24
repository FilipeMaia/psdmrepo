#ifndef PSQT_WDGSPECHIST_H
#define PSQT_WDGSPECHIST_H

#include "ndarray/ndarray.h"
#include <stdint.h> // uint8_t, uint32_t, etc.

#include <PSQt/GUAxes.h>
#include <PSQt/WdgColorBar.h>

#include <QWidget>

//#include <Qt>
#include <QtGui>
#include <QtCore>


namespace PSQt {

// /// @addtogroup PSQt PSQt

/**
 *  @ingroup PSQt
 * 
 *  @brief WdgSpecHist - widget to display spectral historgam for image
 * 
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

class WdgSpecHist : public QWidget
{
 Q_OBJECT // macro is needed for connection of signals and slots

  public:
    WdgSpecHist(QWidget *parent = 0); 
    virtual ~WdgSpecHist();

    PSQt::WdgColorBar* colorBar() { return m_cbar; };
    PSQt::GUAxes* axes() { return m_axes; };

  public slots:
    void onSHistIsFilled(float*, const float&, const float&, const unsigned&);

  private:
    inline const char* _name_(){return "WdgSpecHist";}

    QHBoxLayout* m_hbox;
    QVBoxLayout* m_vbox;
    PSQt::GUAxes* m_axes;
    PSQt::WdgColorBar* m_cbar;
    QGraphicsPathItem* m_path_item;
};

} // namespace PSQt

#endif // PSQT_WDGSPECHIST_H
