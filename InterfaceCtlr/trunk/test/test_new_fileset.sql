--
-- Test new_file_set
-- Run this multiple times to see that new filesets are properly allocated
--
SELECT id from experiment_def AS ed WHERE ed.name = 'First_AMOS' INTO @exper_id;
set @req_bytes = 1000000;
set @newuri = 'temp';
set @newset = 99999;
CALL new_fileset (@exper_id, @req_bytes, @newset, @newuri);

SELECT @newset;
SELECT @newuri;
SELECt * FROM cache;
SELECT * FROM fileset;
