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

//#include <stdio.h> // for C style printf

//----------------------
// Base Class Headers --
//----------------------

#include "RootHist/RootHManager.h"
#include "PSHist/HManager.h"
#include "PSHist/H1.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "root/TRandom.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

using std::cout;
using std::endl;

//using namespace PSTime;

int main ()
{
  cout << "Stars ex_PSHist_1 : main()" << endl; 


  PSHist::HManager *hMan = new RootHist::RootHManager("pshist-test.root", "RECREATE");

  PSHist::H1 *pHis1 = hMan->hist1d( "H1_N0001", "My his1d 1 title", 100, 0., 1.);
  PSHist::H1 *pHis2 = hMan->hist1d( "H1_N0002", "My his1d 2 title", 100, 0., 1.);
  PSHist::H1 *pHis3 = hMan->hist1d( "H1_N0003", "My his1d 3 title", 100, 0., 1.01);

  cout << "Fill histogram" << endl;
	for (int i=0; i<10000; i++)
	  {
            pHis1 -> fill( gRandom->Rndm(1)        ); // Uniform distribution on [0,1]
            pHis2 -> fill( gRandom->Gaus(0.5, 0.1) ); // Gaussian for mu and sigma
            pHis3 -> fill( double(0.0001*i), double(i) );
	  }

  hMan -> write();

  delete hMan;  
}


/*

int garbage()
{

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
  TBranch *pbranch = ptree->Branch("new_v", &new_v, "new_v/F");

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


*/
