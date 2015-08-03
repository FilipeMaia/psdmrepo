//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Test root main functionality which is going to be used in RootHist packadge
//
// Author List:
//      Mikhail Dubrovin
//
//------------------------------------------------------------------------

//-----------------
// C/C++ Headers --
//-----------------

#include "PSHist/Axis.h"

#include <string>
#include <iostream>

using std::cout;
using std::endl;

//using namespace PSTime;

int main ()
{
  double edges[]={0, 0.05, 0.1, 0.2, 0.25, 0.3, 0.5, 0.55, 0.7, 0.9, 1};
  cout << "sizeof(edges) =" << sizeof(edges) << endl;     

  PSHist::Axis *axis1 = new PSHist::Axis((int)10,0,100);  
  axis1->print(std::cout);

  PSHist::Axis *axis2 = new PSHist::Axis(10,edges);  
  axis2->print(std::cout);

  PSHist::Axis axis3(5,edges);  
  axis3.print(std::cout);

  return 0;
}

//-----------------
