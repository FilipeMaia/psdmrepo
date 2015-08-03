


Create missing index (idx) files
********************************

The idx-services creates missing xtc-index (idx) files. It consists of two components: A services that
queries a queue for new requests, submits them to the batch system and checks if the index files have 
been created. The second component are clients that add requests to the queue. The clients are either 
the irods file restore process (univMSSInterface.sh) or a daemon that checks for failed idx transfers 
and adds them to the queue.

The queue is implemented using a local sqlite database. Therefore the clients and server have to run
on the same host.

Tools
=====

idx-q-server 
------------
Service that submits request to batch. Runs as daemon controlled by supervisord. It is configured
using a INI style config file reading the *idxService* section. An example is on the data directory 
of the DmMover package. 


idx-failed-regdb
----------------
Create missing idx files that failed the transfer by a data-mover. It checks the data migration table 
for index files that failed the transfer in the last N hours and the ones that are found are added to the 
queue.

idx-q-cmd 
---------
Command line tool to add request and list status. Access the database that hosts the queue directly. 
The path to the database and the WAIT semaphore files are modified by the two env variables *DM_IDX_DB* 
and *DM_IDX_WAIT*

idx-create
----------
Script that creates an idx file from a xtc file.

Usage: idx_create <xtc-file>

Creates the index files in the index sub-folder of the xtc file. The file is first written to a temp file 
and moved to the proper name if it was successfully created. It will fail if the idx file already exists.

Modules
=======

xtcidxjob.py : simple queue to manage job submission. The jobs create index files.
