

Restore files from tape
=======================

The file restore service consists of three tools:
1. A table (_file_restore_requests_) that holds restore requests and their status
2. A cron job that queries the table for new request and adds it to the iRODS queue. 
3. iRODS that restores the file.



Client commands
---------------


Show all files that have been submitted 
% show_submitted 

dmt-submit
----------



dmt-update-status
-----------------

Check if requests that were submitted to iRODS were restored and mark them as
DONE in the restore table. Requests can be selected by experiment (--exp name|id).
If a restore failed due to a failed transfer from HPSS to disk the file with the
extension _.fromtape_ is left on disk. The _--fix_ option will remove the temp file
and reset the files status to RECEIVED which will cause it to be restored again.


dmt-cntl
--------

show restore request, update the status for a request or add a new request.
