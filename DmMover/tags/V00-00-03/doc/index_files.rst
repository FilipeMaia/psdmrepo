


Create missing index (idx) files
********************************


Tools
=====

Apps
----

idx_q_server : service that submits request to batch
idx_q_cmd  : status of requests and add request for idx creation.
idx_create : create idx file from corresponding idx
 
dm_createIdx_regdb : create missing idx files that failed the transfer by a data-mover

Modules
-------

xtcidxjob.py : simple queue to manage job submission. The jobs create index files.

