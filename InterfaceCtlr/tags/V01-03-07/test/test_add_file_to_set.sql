-- 
-- Test add_file_to_set
-- 
set @exper = 'Chapman2';
set @inst  = 'AMOS';
set @runtype = 'Data';
set @runnum = 345;
set @req_bytes = 1000000;
set @newset = 0;
set @newuri = 'temp';
-- Create a fileset into which we add files.
CALL new_fileset (@exper, @inst, @runtype, @runnum, @req_bytes, @newset, @newuri);
SELECT * FROM fileset;
SELECT @newset;
SELECT @newuri;
set @status = 0;
set @filetype = 'XTC';
CALL add_file_to_set (@newset, @filetype, 'file1', 500000, @status);
SELECT @status;
SELECT * FROM filename;
set @status = 0;
CALL add_file_to_set (@newset, @filetype, 'file2', 500000, @status);
SELECT @status;
SELECT * FROM filename;
set @status = 0;
--
-- This add should fail as we excede fileset size requested.
--
CALL add_file_to_set (@newset, @filetype, 'file3', 5000, @status);
SELECT @status;
SELECT * FROM filename;
