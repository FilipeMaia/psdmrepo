--
-- Test change_fileset_status
--
set @exper = 'ChangeFilenameStatus';
Set @instr = 'XTC'
set @runtype = 'Calibration';
set @runnum = 0123;
set @req_bytes = 1000000;
set @newset = 0;
CALL new_fileset (@exper, @instr, @runtype, @runnum, @req_bytes, @newset, @newuri);
SELECT * FROM fileset;
SELECT @newuri;
set @status = 0;
set @ft = 'EPICS';
CALL add_file_to_set (@newset, @ft, 'file1', 50000, @status);
SELECT @status;
set @status = 0;
set @ft = 'XTC';
CALL add_file_to_set (@newset, @ft, 'file2', 50000, @status);
SELECT @status;
set @status = 0;
CALL add_file_to_set (@newset, @ft, 'file3', 50000, @status);
SELECT @status;
SELECT * FROM files;
SELECT id FROM fileset_status_def AS fs WHERE fs.name = 'Waiting_Translation' INTO @fsstat_id;
set @status = 0;
CALL change_fileset_status (@newset, @fsstat_id, @status);
SELECT * FROM fileset
SELECT * FROM files;
