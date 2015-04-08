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

#include <string>
#include <iostream>
//#include <sstream>

#include <stdio.h>

//----------------------
// Base Class Headers --
//----------------------

#include "root/TROOT.h"
#include "root/TFile.h"
#include "root/TH1D.h"
#include "root/TTree.h"
#include "root/TBranch.h"
#include "root/TRandom.h"
//-------------------------------
// Collaborating Class Headers --
//-------------------------------

//#include "RootHist/....h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using std::cout;
using std::endl;

//using namespace PSTime;

void initialiseRoot() {
  cout << "Initialize root" << endl;
  if( !TROOT::Initialized() ) {
    static TROOT root( "RootManager", "RootManager ROOT God" );
  }
}


int main ()
{
  cout << "Stars main()" << endl; 

  //initialiseRoot();

  TFile *pfile = new TFile("file.root", "RECREATE", "Created for you by RootManager" );
  cout << "Open Root file with name : " << pfile ->GetName() << endl;


  cout << "Create histogram" << endl;
  TH1D *pHis1 = new TH1D("pHis1","My comment to TH1D", 100, 0.5, 100+0.5);


  cout << "Reset and fill histogram" << endl;
        pHis1 -> Reset();
	for (int i=0 ;i<10000;i++)
	  {
            double random = 100 * gRandom->Rndm(1);

	    //pHis1 -> Fill( double(i), 0.1*i );
	    pHis1 -> Fill( random );

	  }


  cout << "Write histogram in file" << endl;
        pHis1 -> Write();


// Define some simple structures
   typedef struct {float x,y,z;} POINT;

   static POINT point;


  cout << "Create tree" << endl;
//TTree *t3 = (TTree*)->Get("t3"); // if tuple existed
  TTree *ptree = new TTree("ptree", "My comment to TTree");


  cout << "Create  a couple of branches" << endl;
  float new_v;
  ptree->Branch("new_v", &new_v, "new_v/F");
  ptree->Branch("point",&point,"x:y:z");


  cout << "Fill branch" << endl;
  for (int i = 0; i < 10000; i++){

      new_v   = gRandom->Gaus(0, 1);

      point.x = gRandom->Gaus(1, 1); 
      point.y = gRandom->Gaus(2, 1); 
      point.z = gRandom->Gaus(3, 1); 

      ptree->Fill();
      //pbranch->Fill();
  }


  cout << "Write tree in file" << endl;
  ptree -> Write();

  
  cout << "Close file" << endl;
  pfile -> Close();


  //cout << "Delete all objects" << endl;
  //delete pHis1;
  //delete ptree;

  return 0;

}
