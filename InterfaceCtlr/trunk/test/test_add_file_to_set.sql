-- 
-- Test add_file_to_set
-- 
SELECT id FROM experiment_def AS ed WHERE ed.name = 'First_AMOS' INTO @exper_id;
set @req_bytes = 1000000;
set @newset = 0;
CALL new_fileset (@exper_id, @req_bytes, @newset, @newuri);
SELECT * FROM fileset;
SELECT @newuri;
set @status = 0;
CALL add_file_to_set (@newset, @exper_id, 'file1', 500000, @status);
SELECT @status;
SELECT * FROM filename;
set @status = 0;
CALL add_file_to_set (@newset, @exper_id, 'file2', 500000, @status);
SELECT @status;
SELECT * FROM filename;
set @status = 0;
CALL add_file_to_set (@newset, @exper_id, 'file3', 5000, @status);
SELECT @status;
SELECT * FROM filename;

