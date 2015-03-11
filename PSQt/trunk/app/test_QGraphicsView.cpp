//=======================

#include <PSQt/GUAxes.h>
#include <PSQt/QGUtils.h> // for PSQt::graph
//#include <PSQt/GURuler.h>
//#include <PSQt/GUView.h>

#include <QtGui/QApplication>

#include <cmath>     // for sin, cos
#include <iostream>  // for std::cout

int main(int argc, char *argv[])
{
  QApplication a(argc, argv);

  float xmin = 0;
  float xmax = 30;
  float ymin = -4;
  float ymax = 12;
  unsigned pbits=1;

  unsigned npts = 1000;
  float x1[npts];
  float y1[npts];
  float y2[npts];
  float y3[npts];
  float dx = (xmax-xmin)/(npts+1);

  unsigned i=0;
  for(float x=xmin; x<xmax; x+=dx, i++) {
    x1[i]   = x;
    y1[i] = 4*sin(x);
    y2[i] = 3*cos(5*x);
    y3[i] = (x>0) ? ymax*sin(x)/x : ymax;
    //i++;
  }

  PSQt::GUAxes* pax1 = new PSQt::GUAxes(0, xmin, xmax, ymin, ymax, pbits);
  //pax1->setGeometry(200, 30, 800, 600);

  PSQt::graph(pax1, x1, y1, npts, "-rT4");
  PSQt::graph(pax1, x1, y2, npts, "-kT");
  PSQt::graph(pax1, x1, y3, npts, "-bT");

  // Show the view
   //pax1->pview()->show();
  pax1->show();

  //QGraphicsView  * view = pax1->pview();
  //QGraphicsScene * view = pax1->pscene();
  //QTransform trans = pax1->transform() 

  return a.exec();
}


