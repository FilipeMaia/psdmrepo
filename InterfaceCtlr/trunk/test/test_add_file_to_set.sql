-- 
-- Test add_file_to_set
-- 
SELECT id FROM experiment_def AS ed WHERE ed.name = 'First_AMOS' INTO @exper_id;
SELECT id FROM filetype_def AS ft WHERE ft.name = 'XTC' INTO @ft_id;
set @req_bytes = 1000000;
set @newset = 0;
SELECT * FROM cache;
-- Create a fileset into which we add files.
CALL new_fileset (@exper_id, @req_bytes, @newset, @newuri);
SELECT * FROM fileset;
SELECT @newuri;
set @status = 0;
CALL add_file_to_set (@newset, @ft_id, 'file1', 500000, @status);
SELECT @status;
SELECT * FROM filename;
set @status = 0;
CALL add_file_to_set (@newset, @ft_id, 'file2', 500000, @status);
SELECT @status;
SELECT * FROM filename;
set @status = 0;
--
-- This add should fail as we excede fileset size requested.
--
CALL add_file_to_set (@newset, @ft_id, 'file3', 5000, @status);
SELECT @status;
SELECT * FROM filename;
