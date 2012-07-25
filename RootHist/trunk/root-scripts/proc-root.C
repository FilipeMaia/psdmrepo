

// To run this script use command: root -q -f proc.C


//void proc(int Nplot=1)
{
  // Settings for pads

   gStyle->SetPadColor(3);
   gStyle->SetPadBorderSize(0);
   gStyle->SetPadBorderMode(0);

   gStyle -> SetTitleXSize(0.05); // set size of axes titles
   gStyle -> SetTitleYSize(0.05);

   //gStyle->SetTitleW(0.35);
   gStyle->SetTitleH(0.1); // set size of the title in top box
 
 
  TFile *f = new TFile("file.root");

  f->ls();
  ptree->Print();

  c1 = new TCanvas("c1","",0,0,500,500);
  c1->Divide(2,3);

  c1->cd(1); pHis1->Draw();
  c1->cd(2); ptree->Draw("new_v");
  c1->cd(3); ptree->Draw("x");
  c1->cd(4); ptree->Draw("y");
  c1->cd(5); ptree->Draw("z");

  gPad -> Update();   

  c1->Print("test-histograms.gif");

  cout << "Sleep for 10 sec..." << endl;
  gSystem->Sleep(10*1000);
  cout << "Wake up!" << endl;

  f -> Close();

}
