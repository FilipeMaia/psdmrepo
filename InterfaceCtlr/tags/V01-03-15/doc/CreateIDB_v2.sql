SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;

-- -----------------------------------------------------
-- Table `active_controller`
-- -----------------------------------------------------
SHOW WARNINGS;

CREATE TABLE IF NOT EXISTS `active_controller` (
  `controller_id` int(10) unsigned DEFAULT NULL COMMENT 'Foreign key of interface_controller process that s active now',
  `updated` timestamp NOT NULL DEFAULT '0000-00-00 00:00:00' ON UPDATE CURRENT_TIMESTAMP COMMENT 'Last update time',
  KEY `active_controller_id_fk` (`controller_id`),
  CONSTRAINT `active_controller_id_fk` FOREIGN KEY (`controller_id`) REFERENCES `interface_controller` (`id`) ON DELETE NO ACTION ON UPDATE NO ACTION
) 
ENGINE=InnoDB
COMMENT='Record of the currently active interface controller.';


-- -----------------------------------------------------
-- Table `translator_node`
-- -----------------------------------------------------
-- DROP TABLE IF EXISTS `translator_node` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `translator_node` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `active` BOOLEAN NOT NULL ,
  `node_uri` TEXT NOT NULL COMMENT 'Initially node:network but may be more general later. The Controller looks in this table to find info for the node on which it is running.' ,
  `description` TEXT NOT NULL COMMENT 'More detailed description of the node.' ,
  PRIMARY KEY (`id`) 
)
ENGINE = InnoDB
COMMENT = 'Contains one row for each translator node.';


SHOW WARNINGS;
-- -----------------------------------------------------
-- Table `node2instr`
-- -----------------------------------------------------
-- DROP TABLE IF EXISTS `node2instr` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `node2instr` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `translator_node_id` INT UNSIGNED NOT NULL,
  `instrument` TINYTEXT NOT NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `node_id_fk`
    FOREIGN KEY (`translator_node_id` )
    REFERENCES `translator_node` (`id` ) ON DELETE NO ACTION ON UPDATE NO ACTION
)
ENGINE = InnoDB
COMMENT = 'Defines allowed instruments for particular node.';


-- -----------------------------------------------------
-- Table `interface_controller`
-- -----------------------------------------------------
-- DROP TABLE IF EXISTS `interface_controller` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `interface_controller` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `fk_translator_node` INT UNSIGNED NOT NULL COMMENT 'Foreign key of translator_cpu on which the translator is running.' ,
  `process_id` INT NOT NULL COMMENT 'Process id of translator process' ,
  `kill_ic` BOOLEAN NOT NULL COMMENT 'Force interface controller process to stop.' ,
  `started` TIMESTAMP NOT NULL DEFAULT '2000-01-01 00:00:00' COMMENT 'Time translator started.' ,
  `stopped` TIMESTAMP NULL DEFAULT NULL COMMENT 'Time translator stopped.' ,
  `log` TEXT COMMENT 'Log file name.' ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `translator_node_fk`
    FOREIGN KEY (`fk_translator_node` )
    REFERENCES `translator_node` (`id` ) ON DELETE NO ACTION ON UPDATE NO ACTION
)
ENGINE = InnoDB
COMMENT = 'Instance of an Interface Controller on a cache_trans_cpu.';


SHOW WARNINGS;
CREATE INDEX `translator_node_fk` ON `interface_controller` (`fk_translator_node` ASC) ;

SHOW WARNINGS;


-- -----------------------------------------------------
-- Table `fileset_status_def`
-- -----------------------------------------------------
-- DROP TABLE IF EXISTS `fileset_status_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `fileset_status_def` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(63) NOT NULL ,
  `description` TEXT NOT NULL ,
  `job_state` varchar(8) DEFAULT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `fileset_name_idx` (`name`)
)
ENGINE = InnoDB
COMMENT = 'Status conditions for a fileset & filenames.';

SHOW WARNINGS;


-- -----------------------------------------------------
-- Table `fileset`
-- -----------------------------------------------------
-- DROP TABLE IF EXISTS `fileset` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `fileset` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `fk_fileset_status` INT UNSIGNED NOT NULL COMMENT 'Foreign key of fileset_status_def indicating status of entire fileset. ' ,
  `experiment` TINYTEXT NOT NULL COMMENT 'Unverified experiment name from online.' ,
  `instrument` TINYTEXT NOT NULL ,
  `run_type` TINYTEXT NOT NULL COMMENT 'Unverified run type e.g. data, calibration etc. from online.' ,
  `run_number` INT UNSIGNED NOT NULL COMMENT 'Unverified run number from online.' ,
  `requested_bytes` BIGINT UNSIGNED NULL COMMENT 'Total bytes requested when fileset is created. ' ,
  `used_bytes` BIGINT UNSIGNED NULL COMMENT 'Presently used bytes.' ,
  `created` TIMESTAMP NOT NULL DEFAULT '2000-01-01 00:00:00' COMMENT 'Date fileset created' ,
  `locked` BOOLEAN NOT NULL COMMENT 'Locked when found by get_fileset_with_status. so only one requestor gets a given fileset. Unlocked whenever the fileset status is changed.' ,
  `priority` int(11) DEFAULT '0',
  `translator_id` int(10) unsigned DEFAULT NULL COMMENT 'ID of the translator processing this fileset.',
  PRIMARY KEY (`id`) ,
  CONSTRAINT `fileset_status_fk`
    FOREIGN KEY (`fk_fileset_status` )
    REFERENCES `fileset_status_def` (`id` ) ON DELETE NO ACTION ON UPDATE NO ACTION,
  CONSTRAINT `translator_id_fk` 
    FOREIGN KEY (`translator_id` ) 
    REFERENCES `translator_process` (`id` )
)
ENGINE = InnoDB
AUTO_INCREMENT=24490
COMMENT = 'A fileset is the container for a set of files.';

SHOW WARNINGS;
CREATE INDEX `fileset_status_fk` ON `fileset` (`fk_fileset_status` ASC) ;
CREATE INDEX `fileset_run_number_idx` ON `fileset` (`run_number`) ;
CREATE INDEX `fileset_experiment_idx` ON `fileset` (`experiment`(255)) ;
CREATE INDEX `fileset_instrument_idx` ON `fileset` (`instrument`(255)) ;
CREATE INDEX `fileset_created_idx` ON `fileset` (`created`) ;
CREATE INDEX `fileset_inst_exp_run_idx` ON `fileset` (`instrument`(255),`experiment`(255),`run_number`) ;
CREATE INDEX `fileset_inst_exp_run_time_idx` ON `fileset` (`instrument`(255),`experiment`(255),`run_number`,`created`) ;

SHOW WARNINGS;


-- -----------------------------------------------------
-- Table `translator_process`
-- -----------------------------------------------------
-- DROP TABLE IF EXISTS `translator_process` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `translator_process` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `fk_fileset` INT UNSIGNED NOT NULL COMMENT 'Foreign key of fileset that''s being translated.' ,
  `kill_tp` BOOLEAN NOT NULL COMMENT 'Force translor process to stop.' ,
  `started` TIMESTAMP NOT NULL DEFAULT '2000-01-01 00:00:00' COMMENT 'Time this process started.' ,
  `stopped` TIMESTAMP NULL COMMENT 'Time this process stopped' ,
  `filesize_bytes` BIGINT UNSIGNED NULL COMMENT 'Translated file size in bytes' ,
  `tstatus_code` INT NULL COMMENT 'Return status code from translator' ,
  `istatus_code` INT NULL COMMENT 'Status code from iRODS put' ,
  `log` TEXT COMMENT 'Log file name.' ,
  `jobid` INT UNSIGNED DEFAULT NULL COMMENT 'ID of the job submitted to batch system.',
  `output_dir` TEXT COMMENT 'Location of the output files for this job.',
  PRIMARY KEY (`id`) ,
  CONSTRAINT `fileset_fk_from_translator`
    FOREIGN KEY (`fk_fileset` )
    REFERENCES `fileset` (`id` ) ON DELETE NO ACTION ON UPDATE NO ACTION
)
ENGINE = InnoDB
AUTO_INCREMENT=22650
COMMENT = 'A record of each instance of the translator we create.';

SHOW WARNINGS;
CREATE INDEX `fileset_fk_from_translator` ON `translator_process` (`fk_fileset` ASC) ;

SHOW WARNINGS;


-- -----------------------------------------------------
-- Table `config_def`
-- -----------------------------------------------------
-- DROP TABLE IF EXISTS `config_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `config_def` (
  `section` TINYTEXT NOT NULL ,
  `param` TINYTEXT NOT NULL ,
  `value` TINYTEXT NOT NULL ,
  `type` ENUM('Integer','Float','String','Date/Time') NOT NULL ,
  `description` TEXT NULL,
  `instrument` TINYTEXT NULL,
  `experiment` TINYTEXT NULL
)
ENGINE = InnoDB
COMMENT = 'Interface DB & controller config params.';

SHOW WARNINGS;


-- -----------------------------------------------------
-- Table `files`
-- -----------------------------------------------------
-- DROP TABLE IF EXISTS `files` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `files` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `fk_fileset_id` INT UNSIGNED NOT NULL COMMENT 'Foreign key of the associated fileset.' ,
  `archive_dir` TEXT COMMENT 'File manager directory where file was archived, NULL if it was not archived.' ,
  `name` TEXT NOT NULL COMMENT 'Full file name including path for nocache option. For the cache option the path is the cache directory.' ,
  `type` ENUM('XTC','EPICS','HDF5') NOT NULL ,
  `size_bytes` BIGINT UNSIGNED NULL ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `fileset_id_fk`
    FOREIGN KEY (`fk_fileset_id` )
    REFERENCES `fileset` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Filenames associated with a fileset.';

SHOW WARNINGS;
CREATE INDEX `fileset_id_fk` ON `files` (`fk_fileset_id` ASC) ;

-- -----------------------------------------------------
-- Table `active_exp`
-- -----------------------------------------------------
-- DROP TABLE IF EXISTS `active_exp` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `active_exp` (
  `instrument` TINYTEXT NOT NULL ,
  `experiment` TINYTEXT NOT NULL ,
  `since` TIMESTAMP NOT NULL ,
  PRIMARY KEY (`instrument`(255),`experiment`(255)) 
)
ENGINE = InnoDB
COMMENT = 'List of experiments which can be processed.';


-- -----------------------------------------------------
-- Data for table `fileset_status_def`
-- do not delete any ID, and do not reuse IDs, any new state
-- should be added at the end with the increasing ID number
-- -----------------------------------------------------

SET AUTOCOMMIT=0;
INSERT INTO fileset_status_def (name, description, job_state) VALUES ('WAIT_FILES', 'Translation request created, waiting for input files', 'QUEUE');
INSERT INTO fileset_status_def (name, description, job_state) VALUES ('WAIT', 'Translation request created, waiting for job submission', 'QUEUE');
INSERT INTO fileset_status_def (name, description, job_state) VALUES ('RUN', 'Translation job is being executed by batch farm', 'RUN');
INSERT INTO fileset_status_def (name, description, job_state) VALUES ('FAIL', 'Translation job execution failed', 'FAIL');
INSERT INTO fileset_status_def (name, description, job_state) VALUES ('FAIL_COPY', 'Failed to copy translated files to output directory', 'FAIL');
INSERT INTO fileset_status_def (name, description, job_state) VALUES ('FAIL_NOINPUT', 'Input data files cannot be located', 'FAIL');
INSERT INTO fileset_status_def (name, description, job_state) VALUES ('FAIL_MKDIR', 'Failed to created directory for output files', 'FAIL');
INSERT INTO fileset_status_def (name, description, job_state) VALUES ('DONE', 'Translation request completed succesfully', 'DONE');
INSERT INTO fileset_status_def (name, description, job_state) VALUES ('PENDING', 'Translator job is submitted, pending in batch queue', 'QUEUE');
INSERT INTO fileset_status_def (name, description, job_state) VALUES ('SUSPENDED', 'Translator job is suspended by batch system', 'RUN');

COMMIT;


INSERT INTO active_controller (controller_id, updated) VALUES (NULL, NULL);
COMMIT;
