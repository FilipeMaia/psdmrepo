
Naming convention
=================

filename = basename of a file
filepath = absolute pathof a file
filedir  = directory that contains file

E.g.:
  filepath = /reg/data/ana01/temp/psdatmgr/amo/amodaq09/xtc/smalldata/e18-r0015-s00-c01.smd.xtc
  filename = /e18-r0015-s00-c01.smd.xtc
  filedir  = /reg/data/ana01/temp/psdatmgr/amo/amodaq09/xtc/smalldata


Testing Data Movers
===================

Create/modify entries in regdb
------------------------------

Show files for an experiemnt (--expid <id>. defaiult 18, amadaq09).

% dmtest_data_migr_status list --status all  | egrep '^18' | column  -t -s  '|'

Set the status for a particular file:

% dmtest_data_migr_status  set  e18-r0015-s00-c01.smd.xtc

 

Test transfers
--------------

% mv2offline-ana --testdb --listonly --onlyone --onetime --nopath -vv --mode local --smd
