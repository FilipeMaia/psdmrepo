--
-- Test change_filename_status
--
SELECT id FROM experiment_def AS ed WHERE ed.name = 'First_AMOS' INTO @exper_id;
SELECT id FROM filetype_def AS ft WHERE ft.name = 'XTC' INTO @ft_id;
set @req_bytes = 1000000;
set @newset = 0;
CALL new_fileset (@exper_id, @req_bytes, @newset, @newuri);
SELECT * FROM fileset;
SELECT @newuri;
set @status = 0;
CALL add_file_to_set (@newset, @ft_id, 'file1', 50000, @status);
SELECT @status;
set @status = 0;
CALL add_file_to_set (@newset, @ft_id, 'file2', 50000, @status);
SELECT @status;
set @status = 0;
CALL add_file_to_set (@newset, @ft_id, 'file3', 50000, @status);
SELECT @status;
SELECT * FROM filename;
SELECT id FROM fileset_status_def AS fs WHERE fs.name = 'Translation_Failed' INTO @fsstat_id;
SELECT @fsstat_id;
set @status = 0;
SELECT id FROM filename AS fn WHERE fn.filename = 'file2' INTO @fn_id;
SELECT * FROM filename;
CALL change_filename_status (@fn_id, @fsstat_id, @status);
SELECT @status;
SELECT * FROM fileset;
SELECT * FROM filename;