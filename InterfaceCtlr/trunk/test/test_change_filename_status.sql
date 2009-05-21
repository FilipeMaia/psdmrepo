--
-- Test change_filename_status
--
SELECT id FROM experiment_def AS ed WHERE ed.name = 'First_AMOS' INTO @exper_id;
set @req_bytes = 1000000;
set @newset = 0;
CALL new_fileset (@exper_id, @req_bytes, @newset, @newuri);
SELECT * FROM fileset;
SELECT @newuri;
SELECT id FROM filename_status_def AS fs WHERE fs.name = 'Waiting_Translation' INTO @fsstat_id;
set @status = 0;
SELECT * FROM filename;
CALL change_filename_status (@newset, @fsstat_id);
SELECT * FROM filename;
