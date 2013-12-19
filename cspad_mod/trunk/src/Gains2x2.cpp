//--------------------------------------------------------------------------
// File and Version Information:
// 	$Id$
//
// Description:
//	Class Gains2x2...
//
// Author List:
//      Philip Hart
//
//------------------------------------------------------------------------

//-----------------------
// This Class's Header --
//-----------------------
#include "cspad_mod/Gains2x2.h"

//-----------------
// C/C++ Headers --
//-----------------

//-------------------------------
// Collaborating Class Headers --
//-------------------------------
#include "MsgLogger/MsgLogger.h"
#include "PSEvt/EventId.h"

#include "root/TROOT.h"
#include "root/TApplication.h"
#include "root/TFile.h"
#include "root/TF1.h"

//-----------------------------------------------------------------------
// Local Macros, Typedefs, Structures, Unions and Forward Declarations --
//-----------------------------------------------------------------------

// This declares this class as psana module
using namespace cspad_mod;

static TFile* rootFile;
//static TTree* ntup;

PSANA_MODULE_FACTORY(Gains2x2)

//		----------------------------------------
// 		-- Public Function Member Definitions --
//		----------------------------------------
  
namespace cspad_mod {

//----------------
// Constructors --
//----------------
Gains2x2::Gains2x2 (const std::string& name)
  : Module(name)
  , m_src()
  , m_maxEvents()
  , m_filter()
  , m_count(0)
  , m_calibKey()
  , m_minClusterE()
  , m_maxSingleNeighborE()
  , m_gainFile()
  , m_rootFile()
{
  // get the values from configuration or use defaults
  m_src = configSrc("source", "DetInfo(:Cspad2x2)");
  m_maxEvents = config("events", 32U);
  m_filter = config("filter", false);
  m_calibKey = configStr("inputKey", "");
  m_minClusterE = config("minClusterE", 15.);
  m_maxSingleNeighborE = config("maxSingleNeighborE", 6.);
  m_gainFile = configStr("gainFile", "");
  m_rootFile = configStr("rootFile", "");

  _quads = 1;
  _twoXones = 2;

}

//--------------
// Destructor --
//--------------
Gains2x2::~Gains2x2 ()
{
}

/// Method which is called once at the beginning of the job
void 
Gains2x2::beginJob(Event& evt, Env& env)
{
  const char* rf = m_rootFile.c_str();
  printf("making root file %s\n", rf);
  rootFile = new TFile(rf, "RECREATE");

  char levelName[80];
  char plotTitle[80], plotName[80];
  TDirectory* detDir;
  TDirectory* tXoDir;
  TDirectory* clusterDir;
  TDirectory* bankDir;
  sprintf(levelName, "%d", 666);// was devInfo # for det from online 
  printf("make dir %s\n", levelName);
  detDir = rootFile->mkdir(levelName);
  detDir->cd();

  for (unsigned quad=0; quad<_quads; quad++) {
    //    if (!populatedQuadrant(quad)) continue;
    for (unsigned twoXone=0; twoXone<_twoXones; twoXone++) {
      sprintf(levelName, "q%d_t%02d",quad, twoXone);
      tXoDir = detDir->mkdir(levelName);
      tXoDir->cd();

      sprintf(plotName, "fnSub_broad_q%1d_%1ds", quad, twoXone);
      sprintf(plotTitle, "Quad %d 2x1 %d pixel dist, peak frame mean subtracted pixel data", quad, twoXone);
      pixelPeakSub[quad][twoXone]   = new TH1D(plotName, plotTitle, 1000, -50, 950);
      sprintf(plotName, "fnSub_broad_q%1d_%1ds_singles", quad, twoXone);
      sprintf(plotTitle, "Quad %d 2x1 %d single pixel cluster dist, peak frame mean subtracted pixel data", quad, twoXone);
      pixelPeakSubSingles[quad][twoXone] = new TH1D(plotName, plotTitle, 1000, -50, 950);
      
      clusterDir = tXoDir->mkdir("pixels");
      clusterDir->cd();
      
      for (unsigned row=0; row < ROWS; row++) {
	if (row%26==0) {
	  char bankName[16];
	  sprintf(bankName, "bank_%d_%02d", row/194, (row%194)/26);
	  bankDir = clusterDir->mkdir(bankName);
	  bankDir->cd();
	}
	for (unsigned col=0; col < COLS; col++) {
	  if(true) { //fillAllPixels and fitPixel(quad, twoXone, col, row)) {
	    sprintf(plotName, "singlePhotonPeak_q%d_s%d_c%03d_r%03d", quad, twoXone, col, row);
	    sprintf(plotTitle, "Quad %d 2x1 %d col %d row %d pixel dist, peak frame mean subtracted pixel data", quad, twoXone, col, row);
	    if(true) {//!fillWithRaw) {
	      singlePhotonPeak[quad][twoXone][col][row] = new TH1I(plotName, plotTitle, 150, -25, 125);
	    } else {
	      singlePhotonPeak[quad][twoXone][col][row] = new TH1I(plotName, plotTitle, 200, -25, 175);
	    }
	  }
	}
      }

      clusterDir = tXoDir->mkdir("singles");
      clusterDir->cd();
      if(true) {
	for (unsigned row=0; row < ROWS; row++) {
	  if (row%26==0) {
	    char bankName[16];
	    sprintf(bankName, "bank_%d_%02d", row/194, (row%194)/26);
	    bankDir = clusterDir->mkdir(bankName);
	    bankDir->cd();
	  }
	  for (unsigned col=0; col < COLS; col++) {
	    if(true) {//fillAllPixels and fitPixel(quad, twoXone, col, row)) {
	    sprintf(plotName, "singlePixelCluster_q%d_s%d_c%03d_r%03d", quad, twoXone, col, row);
	    sprintf(plotTitle, "Quad %d 2x1 %d col %d row %d single pixel cluster dist, peak frame mean subtracted pixel data", quad, twoXone, col, row);
	      singlePixelCluster[quad][twoXone][col][row] = new TH1I(plotName, plotTitle, 200, -25, 175);
	    }
	  }
	}
      }
      
      clusterDir = tXoDir->mkdir("doubles");
      clusterDir->cd();
      if(true) {
	for (unsigned row=0; row < ROWS; row++) {
	  if (row%Psana::CsPad2x2::RowsPerBank==0) {
	    char bankName[16];
	    sprintf(bankName, "bank_%d_%02d", row/194, (row%194)/Psana::CsPad2x2::RowsPerBank);
	    bankDir = clusterDir->mkdir(bankName);
	    bankDir->cd();
	  }
	  for (unsigned col=0; col < COLS; col++) {
	    if(true) {//fillAllPixels and fitPixel(quad, twoXone, col, row)) {
	      sprintf(plotName, "doublePixelCluster_q%d_s%d_c%03d_r%03d", quad, twoXone, col, row);
	      sprintf(plotTitle, "Quad %d 2x1 %d col %d row %d two pixel cluster dist, peak frame mean subtracted pixel data", quad, twoXone, col, row);
	      doublePixelCluster[quad][twoXone][col][row] = new TH1I(plotName, plotTitle, 200, -25, 175);
	    }
	  }
	}
      }
    }
  }

}

/// Method which is called at the beginning of the run
void 
Gains2x2::beginRun(Event& evt, Env& env)
{
}

/// Method which is called at the beginning of the calibration cycle
void 
Gains2x2::beginCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called with event data, this is the only required 
/// method, all other methods are optional
void 
Gains2x2::event(Event& evt, Env& env)
{
  // this is how to gracefully stop analysis job
  if (m_count%1000==0) printf("event %lu\n", m_count);
  if (m_count >= m_maxEvents) stop();

  shared_ptr<Psana::CsPad2x2::ElementV1> cspadData = evt.get(m_src, m_calibKey);
  if (! cspadData) {
    printf("error finding 140k\n");
    return;
    //    stop();
  }

  /*
  printf("%f %f\n", 
	 cspadData->common_mode(0), 
	 cspadData->common_mode(1));// this is by section
  */
  const ndarray<const int16_t, 3>& data = cspadData->data();
  const int rOffset[4] = {0, -1, 1, 0};
  const int cOffset[4] = {-1, 0, 0, 1};

  for (unsigned q=0; q<_quads; q++) {
    for (unsigned c=0; c<COLS; c++) {
      for (unsigned r=0; r<ROWS; r++) {
	unsigned nNeighbors[2] = {0, 0};
	float clusterE[2];
	for (unsigned s=0; s<2; s++) {// this is efficient for the 140k xtc ordering
	  clusterE[s] = data[c][r][s];
	  //	  printf("s %d c %d r %d e %f\n", s, c, r, clusterE[s]);
	  pixelPeakSub[q][s]->Fill(clusterE[s]);
	  singlePhotonPeak[q][s][c][r]->Fill(clusterE[s]);
	  if (r==0 or c==0 or r==(ROWS-1) or c==(COLS-1)) continue; // probably reorder
	  // maybe in two sets of loops
	  if (clusterE[s]<m_minClusterE) continue;
	  for (unsigned i=0;i<4; i++) {
	    //	    float tmp = data[r+rOffset[i]][c+cOffset[i]][s];
	    float tmp = data[c+cOffset[i]][r+rOffset[i]][s];
	    if (tmp>m_maxSingleNeighborE) {
	      nNeighbors[s] += 1;
	      clusterE[s] += tmp;
	    }
	  }
	  if (nNeighbors[s]==0) {
	    pixelPeakSubSingles[q][s]->Fill(clusterE[s]);
	    singlePixelCluster[q][s][c][r]->Fill(clusterE[s]);
	  } else if (nNeighbors[2]==1) {
	    doublePixelCluster[q][s][c][r]->Fill(clusterE[s]);
	  }
	}
      }
    }
  }

  // increment event counter
  ++ m_count;
}
  
/// Method which is called at the end of the calibration cycle
void 
Gains2x2::endCalibCycle(Event& evt, Env& env)
{
}

/// Method which is called at the end of the run
void 
Gains2x2::endRun(Event& evt, Env& env)
{
}

/// Method which is called once at the end of the job
void 
Gains2x2::endJob(Event& evt, Env& env)
{

  if (m_gainFile != "") {
    FILE* gf = fopen(m_gainFile.c_str(), "w");
    if (gf) {
      float par[3], parSingles[3], chi2PerDOF;
      for (unsigned q=0; q<_quads; q++) {
	for (unsigned c=0; c<COLS; c++) {
	  for (unsigned r=0; r<ROWS; r++) {
	    for (unsigned s=0; s<2; s++) {
	      memset(par, 0, sizeof(float)*3);
	      memset(parSingles, 0, sizeof(float)*3);
	      chi2PerDOF = 0.;

	      singlePhotonPeak[q][s][c][r]->Fit("gaus", "Q");
	      singlePixelCluster[q][s][c][r]->Fit("gaus", "Q");
	      TF1 *g0 = singlePhotonPeak[q][s][c][r]->GetFunction("gaus");
	      TF1 *g1 = singlePixelCluster[q][s][c][r]->GetFunction("gaus");
	      
	      if(g0) {
		par[0] = g0->GetParameter(0);
		par[1] = g0->GetParameter(1);
		par[2] = g0->GetParameter(2);
		if (not g1) {
		  singlePhotonPeak[q][s][c][r]->Fit("gaus", "Q", "", m_minClusterE, m_minClusterE*5); // assumes sensible min cluster E and histogram range
		  g1 = singlePhotonPeak[q][s][c][r]->GetFunction("gaus");
		}
		if (g1) {
		  parSingles[0] = g1->GetParameter(0);
		  parSingles[1] = g1->GetParameter(1);
		  parSingles[2] = g1->GetParameter(2);
		  chi2PerDOF = g1->GetChisquare()/g1->GetNDF();
		}
	      }
	      fprintf(gf, "%f %f %f %f %f %f %f\n", 
		      par[0], par[1], par[2],
		      parSingles[0], parSingles[1], parSingles[2],
		      chi2PerDOF);
	    }
	  }
	}
      }
    }
  }
	  
  printf("closing root file\n");
  rootFile->Write();
  rootFile->Close();
}

} // namespace cspad_mod
