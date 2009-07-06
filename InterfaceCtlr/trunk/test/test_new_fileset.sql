--
-- Test new_file_set
-- Run this multiple times to see that new filesets are properly allocated
--
set @exper = 'AMOSexper';
set @instr = 'AMOS';
set @runtype = 'Calibration';
set @runnum = 123;
set @req_bytes = 1000000;
set @newuri = 'temp';
set @newset = 99999;
SELECT * from cache;
CALL new_fileset (@exper, @inst, @runtype, @runnum, @req_bytes, @newset, @newuri);

SELECT @newset;
SELECT @newuri;
SELECt * FROM cache;
SELECT * FROM fileset;
