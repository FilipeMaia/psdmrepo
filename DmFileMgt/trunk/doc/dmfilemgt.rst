



Move experiment folders
=======================

The _dm_mv_exp_dir_ is a tool to move an experiment folder (xtc,hdf5 or usr) from its current 
ana filesystem to a new one. 

Usage
-----

Create folders in the new location, create links for files in the new location and update the 
folder link in the experiment nfs directory.

% dm_mv_exp_dir --setup --link cxi cxi12345 xtc ana12 ana01 

Now start the transfer and after a file has been copied to the new location remove the old one

% dm_mv_exp_dir --trans --clean cxi cxi12345 xtc ana12 ana01 

