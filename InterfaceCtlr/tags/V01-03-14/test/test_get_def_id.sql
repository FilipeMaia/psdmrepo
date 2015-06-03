-- 
-- Test get_def_id - Later when dynamic SQL works
-- 
set @deftbl = 'experiment_def';
set @defname= 'First_AMOS';
CALL get_def_id (@deftbl, @defname, @defid);
SELECT @defid;