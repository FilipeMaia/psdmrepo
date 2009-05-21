--
-- Test new_file_set
--
set @exper_id = 1;
set @req_bytes = 1000000;
set @newuri = 'temp';
set @newset = 99;
CALL new_fileset (@exper_id, @req_bytes, @newset, @newuri);

SELECT @newset;
SELECT @newuri;
SELECt * FROM cache;
SELECT * FROM fileset;
 