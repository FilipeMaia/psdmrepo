-- 
-- Test nocache stored procedures
-- 
set @exper = 'Chapman1';
set @inst  = 'AMOS';
set @runtype = 'Data';
set @runnum = 0345;
set @newset = 0;
-- Create a fileset into which we add files.
CALL new_fileset_nocache (@exper, @inst, @runtype, @runnum, @newset);
SELECT * FROM fileset;
SELECT @newset;
set @status = 0;
set @filetype = 'XTC';
CALL add_file_nocache (@newset, @filetype, '/data/rcs/cache/0000.xtc', @status);
SELECT @status;
SELECT * FROM files;
set @status = 0;
CALL add_file_nocache (@newset, @filetype, '/data/rcs/cache/0001.xtc', @status);
SELECT @status;
SELECT * FROM files;
set @status = 0;
--
-- This add should fail as the file already exists.
--
CALL add_file_nocache (@newset, @filetype, '/data/rcs/cache/0001.xtc',@status);
SELECT @status;
SELECT * FROM files;
