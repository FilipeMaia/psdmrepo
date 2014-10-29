


Create missing index (idx) files
********************************

The idx-services creates missing xtc-index (idx) files. It consists of two components: A services that
queries a queue for new requests, submits them to the batch queue and checks if the index files have 
been created. The second component are clients that add requests to the queue. The clients are either 
the irods file restore process (univMSSInterface.sh) or a daemon that checks for failed idx transfers 
and adds them to the queue.

Tools
=====

idx_q_server 
------------
Service that submits request to batch. Runs as daemon controlled by supervisord.

idx_failed_regdb
----------------
create missing idx files that failed the transfer by a data-mover

idx_q_cmd 
---------
command line tool to add request and list status.

idx_create
----------
Script that creates an idx file from a xtc file.

Usage: idx_create <xtc-file>

Creates the index files in the index sub-folder of the xtc file. The file is first written to a temp file 
and moved to the proper name if it was successfully created. It will fail if the idx file already exists.

Modules
=======

xtcidxjob.py : simple queue to manage job submission. The jobs create index files.
