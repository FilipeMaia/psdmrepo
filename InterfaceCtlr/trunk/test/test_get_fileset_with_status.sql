--
-- Test get_fileset_with_status
-- This also assumes that change_fileset_status works
--
set @exper = 'TEST getfileset';
set @instr = 'AMOS';
set @runtype = 'Data';
set @runnum = 0432;
set @req_bytes = 1000000;
set @newset = 0;
set @getset = 0;
-- Create several filesets to insure we get the oldest that matches
CALL new_fileset (@exper, @instr, @runtype, @runnum, @req_bytes, @newset, @newuri);
CALL new_fileset (@exper, @instr, @runtype, @runnum, @req_bytes, @newset, @newuri);
CALL new_fileset (@exper, @instr, @runtype, @runnum, @req_bytes, @newset, @newuri);
SELECT * FROM fileset;
SELECT @newset;
SELECT @newuri;
SELECT id FROM fileset_status_def AS fd WHERE fd.name = "Initial_Entry" INTO @stat_id;
SELECT @stat_id;
CALL get_fileset_with_status (@stat_id, @getset);
SELECT @getset;
SELECT id FROM fileset_status_def AS fd WHERE fd.name = "Waiting_Translation" INTO @stat_id;
CALL get_fileset_with_status (@stat_id, @getset);
SELECT @getset;
CALL change_fileset_status (@newset, @stat_id, @status);
SELECT @status;
SELECT * FROM fileset;
CALL get_fileset_with_status (@stat_id, @getset);
SELECT @getset;

