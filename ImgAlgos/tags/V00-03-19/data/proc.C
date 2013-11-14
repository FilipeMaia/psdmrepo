
// Run this script by the command 
// root -q -f proc.C

//void proc(int Nplot=1)
{
  // Settings for good style
   gStyle -> SetPadColor(3);
   gStyle -> SetPadBorderSize(0);
   gStyle -> SetPadBorderMode(0);
   gStyle -> SetTitleXSize(0.05); // set size of axes titles
   gStyle -> SetTitleYSize(0.05);
   gStyle -> SetTitleH(0.1); // set size of the title in top box

  TFile *f = new TFile("file.root");

  f->ls();
  peakTuple->Print();


  TH1D* hNpix = new TH1D("hNpix","Number of pixels in peak", 20, 0.5, 20.5);
  TH1D* hQuad = new TH1D("hQuad","Quad", 4, -0.5, 3.5);
  TH1D* hSect = new TH1D("hSect","Sect", 8, -0.5, 7.5);

  TString ftype("png"); // png, pdf, eps, gif, svg
  TString fname("histos");

  c1 = new TCanvas("c1","",0,0,600,800);
  c1->Divide(2,3);

  c1->cd(1); hNpeaks->Draw();
  c1->cd(2); peakTuple->Draw("npix>>hNpix"); 
  c1->cd(3); peakTuple->Draw("atot");
  c1->cd(4); peakTuple->Draw("btot");
  c1->cd(5); peakTuple->Draw("noise","noise<400");
  c1->cd(6); peakTuple->Draw("son","son<1000");

  gPad -> Update();
  c1->Print(fname+"_p1."+ftype);

  c2 = new TCanvas("c2","",600,0,600,800);
  c2->Divide(2,3);

  c2->cd(1); peakTuple->Draw("quad>>hQuad");
  c2->cd(2); peakTuple->Draw("sect>>hSect");
  c2->cd(3); peakTuple->Draw("col");
  c2->cd(4); peakTuple->Draw("row");
  c2->cd(5); peakTuple->Draw("sigma_c");
  c2->cd(6); peakTuple->Draw("sigma_r");

  gPad -> Update();
  c2->Print(fname+"_p2."+ftype);

  cout << "Sleep for 10 sec..." << endl;
  gSystem->Sleep(10*1000);
  cout << "Wake up and exit root." << endl;

  f -> Close();
}
