--
-- Test change_filename_status
--
set @exper = 'ChangeFilenameStatus';
set @inst  = 'AMOS'
set @runtype = 'Calibration';
set @runnum = 123;
set @req_bytes = 1000000;
set @newset = 0;
set @newuri = 'temp';
CALL new_fileset (@exper, @inst, @runtype, @runnum, @req_bytes, @newset, @newuri);
SELECT * FROM fileset;
SELECT @newuri;
SELECT @newset;
set @status = 0;
set @ft = 'XTC';
CALL add_file_to_set (@newset, @ft, 'file1', 50000, @status);
SELECT @status;
set @status = 0;
set @ft = 'EPICS';
CALL add_file_to_set (@newset, @ft, 'file2', 50000, @status);
SELECT @status;
set @status = 0;
CALL add_file_to_set (@newset, @ft, 'file3', 50000, @status);
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