

// To run this script use command: root -q -f proc-pshist.C


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
 
 
  TFile *f = new TFile("pshist-test.root");

  f->ls();

  //------------------------------------

  c1 = new TCanvas("c1","",0,0,800,800);
  c1->Divide(2,2);

  c1->cd(1); H1_N0001->Draw();
  c1->cd(2); H1_N0002->Draw();
  c1->cd(3); H1_N0003->Draw();
  c1->cd(4); H1_N0004->Draw();

  gPad -> Update();

  //------------------------------------

  c2 = new TCanvas("c2","",200,0,800,800);
  c2->Divide(2,2);

  c2->cd(1); H2_N0001->Draw("box");
  c2->cd(2); H2_N0002->Draw("COLORZ");
  c2->cd(3); H2_N0003->Draw("box");
  c2->cd(4); H2_N0004->Draw("COLORZ");

  gPad -> Update();   

  //------------------------------------

  c3 = new TCanvas("c3","",400,0,800,800);
  c3->Divide(2,2);

  c3->cd(1); P1_N0001->Draw();
  c3->cd(2); P1_N0002->Draw();
  c3->cd(3); P1_N0003->Draw();
  c3->cd(4); P1_N0004->Draw();

  gPad -> Update();

  //------------------------------------

  c4 = new TCanvas("c4","",600,0,800,800);
  c4->Divide(2,2);

             TUPLE_N1->Print();
  c4->cd(1); TUPLE_N1->Draw("EBEAM");
  c4->cd(2); TUPLE_N1->Draw("Freq");
  c4->cd(3); TUPLE_N1->Draw("x");
  c4->cd(4); TUPLE_N1->Draw("y");

  gPad -> Update();

  //------------------------------------

  c1 -> Print("plot1.gif"); // works for gif, eps, etc.
  c2 -> Print("plot2.gif");
  c3 -> Print("plot3.gif");
  c4 -> Print("plot4.gif");

  //------------------------------------

  cout << "Sleep for 10 sec..." << endl;
  gSystem->Sleep(10*1000);
  cout << "Wake up, and exit root." << endl;


  f -> Close();

}
