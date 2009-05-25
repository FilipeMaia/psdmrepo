--
-- Test get_fileset_with_status
-- This also assumes that change_fileset_status works
--
SELECT id FROM experiment_def AS ed WHERE ed.name = 'First_AMOS' INTO @exper_id;
set @req_bytes = 1000000;
set @newset = 0;
CALL new_fileset (@exper_id, @req_bytes, @newset, @newuri);
SELECT * FROM fileset;
SELECT @newuri;
SELECT id FROM fileset_Status_def AS fd WHERE fd.name = "Initial_Entry" INTO @stat_id;
CALL get_fileset_with_status (@stat_id, @exper_id, @getset);
SELECT @getset;
SELECT id FROM fileset_Status_def AS fd WHERE fd.name = "Waiting_Translation" INTO @stat_id;
CALL get_fileset_with_status (@stat_id, @exper_id, @getset);
SELECT @getset;
CALL change_fileset_status (@newset, @stat_id, @status);
SELECT @status;
CALL get_fileset_with_status (@stat_id, @exper_id, @getset);
SELECT @getset;
SELECT * from fileset;

