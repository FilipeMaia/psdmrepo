set @autocommit = 0;

--
-- drop some useless tables 
--
DROP TABLE IF EXISTS archive_files;
DROP TABLE IF EXISTS archive_fileset;
DROP TABLE IF EXISTS archive_interface_controller;
DROP TABLE IF EXISTS archive_translator_node;
DROP TABLE IF EXISTS archive_translator_process;

DROP TABLE IF EXISTS status_from_offline;
DROP TABLE IF EXISTS offline_status_msg_def;
DROP TABLE IF EXISTS status_from_online;
DROP TABLE IF EXISTS online_status_msg_def;
DROP TABLE IF EXISTS cache;

--
-- change translator table, add batch job ID and output directory column
--
ALTER TABLE translator_process ADD COLUMN ( 
	`jobid` INT UNSIGNED DEFAULT NULL COMMENT 'ID of the job submitted to batch system.',
	`output_dir` TEXT DEFAULT NULL COMMENT 'Location of the output files for this job.'
);

--
-- change fileset table, translator id column
--
ALTER TABLE fileset ADD COLUMN ( 
	`translator_id` INT UNSIGNED DEFAULT NULL COMMENT 'ID of the translator processing this fileset.',
	CONSTRAINT `translator_id_fk` FOREIGN KEY (`translator_id` ) REFERENCES `translator_process` (`id` )	
);

--
-- remove unused columns
--
ALTER TABLE translator_process 
	DROP COLUMN `tru_utime`,
	DROP COLUMN `tru_stime`,
	DROP COLUMN `tru_maxrss`,
	DROP COLUMN `tru_ixrss`,
	DROP COLUMN `tru_idrss`,
	DROP COLUMN `tru_isrss`,
	DROP COLUMN `tru_minflt`,
	DROP COLUMN `tru_majflt`,
	DROP COLUMN `tru_nswap`,
	DROP COLUMN `tru_inblock`,
	DROP COLUMN `tru_outblock`,
	DROP COLUMN `tru_msgsnd`,
	DROP COLUMN `tru_msgrcv`,
	DROP COLUMN `tru_nsignals`,
	DROP COLUMN `tru_nvcsw`,
	DROP COLUMN `tru_nivcsw`;

ALTER TABLE fileset DROP COLUMN `fk_cache`;

ALTER TABLE translator_node 
	DROP COLUMN `translate_uri`,
	DROP COLUMN `log_uri`;

--
-- Translator now is not tied to controller
--
ALTER TABLE translator_process 
	DROP FOREIGN KEY `interface_controller_fk`,
	DROP KEY `interface_controller_fk`,
	DROP COLUMN `fk_interface_controller`;

--
-- new table for active controller instance
--
DROP TABLE IF EXISTS `active_controller`;
CREATE TABLE `active_controller` (
  `controller_id` INT UNSIGNED DEFAULT NULL COMMENT 'Foreign key of interface_controller process that s active now' ,
  `updated` TIMESTAMP DEFAULT 0 ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update time',
  CONSTRAINT `active_controller_id_fk`
    FOREIGN KEY (`controller_id` )
    REFERENCES `interface_controller` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION )
ENGINE = InnoDB
COMMENT = 'Record of the currently active interface controller.';

--
-- Table contains exactly one record
--
INSERT INTO `active_controller` (`controller_id`) VALUES (NULL);

--
-- rename status 
--
ALTER TABLE fileset_status_def ADD COLUMN (`job_state` varchar(8));
DELETE FROM fileset_status_def WHERE name = 'Being_Copied';
DELETE FROM fileset_status_def WHERE name = 'Archived_OK';
DELETE FROM fileset_status_def WHERE name = 'Translation_Complete';
UPDATE fileset_status_def 
	SET name = 'WAIT_FILES', job_state = 'QUEUE', description = 'Translation request created, waiting for input files' 
	WHERE name = 'Initial_Entry';
UPDATE fileset_status_def 
	SET name = 'WAIT', job_state = 'QUEUE', description = 'Translation request created, waiting for job submission' 
	WHERE name = 'Waiting_Translation';
UPDATE fileset_status_def 
	SET name = 'RUN', job_state = 'RUN', description = 'Translation job is being executed by batch farm' 
	WHERE name = 'Being_Translated';
UPDATE fileset_status_def 
	SET name = 'DONE', job_state = 'DONE', description = 'Translation request completed succesfully' 
	WHERE name = 'Complete';
UPDATE fileset_status_def 
	SET name = 'FAIL', job_state = 'FAIL', description = 'Translation job execution failed' 
	WHERE name = 'Translation_Error';
UPDATE fileset_status_def 
	SET name = 'FAIL_MKDIR', job_state = 'FAIL', description = 'Failed to created directory for output files' 
	WHERE name = 'H5Dir_Error';
UPDATE fileset_status_def 
	SET name = 'FAIL_NOINPUT', job_state = 'FAIL', description = 'Input data files cannot be located' 
	WHERE name = 'Empty_Fileset';
UPDATE fileset_status_def 
	SET name = 'FAIL_COPY', job_state = 'FAIL', description = 'Failed to copy translated files to output directory' 
	WHERE name = 'Archive_Error';
INSERT INTO fileset_status_def (name, description, job_state) 
	VALUES ('PENDING', 'Translator job is submitted, pending in batch queue', 'QUEUE');
INSERT INTO fileset_status_def (name, description, job_state) 
	VALUES ('SUSPENDED', 'Translator job is suspended by batch system', 'RUN');
