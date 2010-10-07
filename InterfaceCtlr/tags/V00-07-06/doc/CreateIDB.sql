SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';


-- -----------------------------------------------------
-- Table `translator_node`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `translator_node` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `translator_node` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `active` BOOLEAN NOT NULL ,
  `node_uri` TEXT NOT NULL COMMENT 'Initially node:network but may be more general later. The Controller looks in this table to find info for the node on which it is running.' ,
  `translate_uri` TEXT NOT NULL COMMENT 'Local directory uri for translated files' ,
  `log_uri` TEXT NOT NULL COMMENT 'Local directory uri for log files' ,
  `description` TEXT NOT NULL COMMENT 'More detailed description of the node.' ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'Contains one row for each translator node.';

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `node2instr`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `node2instr` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `node2instr` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `translator_node_id` INT UNSIGNED NOT NULL,
  `instrument` TINYTEXT NOT NULL,
  PRIMARY KEY (`id`),
  CONSTRAINT `node_id_fk`
    FOREIGN KEY (`translator_node_id` )
    REFERENCES `interface_db`.`translator_node` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Defines allowed instruments for particular node.';

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `interface_controller`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `interface_controller` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `interface_controller` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `fk_translator_node` INT UNSIGNED NOT NULL COMMENT 'Foreign key of translator_cpu on which the translator is running.' ,
  `process_id` INT NOT NULL COMMENT 'Process id of translator process' ,
  `kill_ic` BOOLEAN NOT NULL COMMENT 'Force interface controller process to stop.' ,
  `started` TIMESTAMP NOT NULL DEFAULT '2000-01-01 00:00:00' COMMENT 'Time translator started.' ,
  `stopped` TIMESTAMP NULL COMMENT 'Time translator stopped.' ,
  `log` TEXT COMMENT 'Log file name.`,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `translator_node_fk`
    FOREIGN KEY (`fk_translator_node` )
    REFERENCES `interface_db`.`translator_node` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Instance of an Interface Controller on a cache_trans_cpu.';

SHOW WARNINGS;
CREATE INDEX `translator_node_fk` ON `interface_controller` (`fk_translator_node` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `fileset_status_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `fileset_status_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `fileset_status_def` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(63) NOT NULL ,
  `description` TEXT NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'Status conditions for a fileset & filenames.';

SHOW WARNINGS;
CREATE UNIQUE INDEX `fileset_name_idx` ON `fileset_status_def` (`name` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `fileset`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `fileset` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `fileset` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `fk_cache` INT UNSIGNED NULL COMMENT 'Foreign key of directory uri where files of this fileset reside.' ,
  `fk_fileset_status` INT UNSIGNED NOT NULL COMMENT 'Foreign key of fileset_status_def indicating status of entire fileset. ' ,
  `experiment` TINYTEXT NOT NULL COMMENT 'Unverified experiment name from online.' ,
  `instrument` TINYTEXT NOT NULL ,
  `run_type` TINYTEXT NOT NULL COMMENT 'Unverified run type e.g. data, calibration etc. from online.' ,
  `run_number` INT UNSIGNED NOT NULL COMMENT 'Unverified run number from online.' ,
  `requested_bytes` BIGINT UNSIGNED NULL COMMENT 'Total bytes requested when fileset is created. ' ,
  `used_bytes` BIGINT UNSIGNED NULL COMMENT 'Presently used bytes.' ,
  `created` TIMESTAMP NOT NULL COMMENT 'Date fileset created' ,
  `locked` BOOLEAN NOT NULL COMMENT 'Locked when found by get_fileset_with_status. so only one requestor gets a given fileset. Unlocked whenever the fileset status is changed.' ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `fileset_status_fk`
    FOREIGN KEY (`fk_fileset_status` )
    REFERENCES `interface_db`.`fileset_status_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
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
DROP TABLE IF EXISTS `translator_process` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `translator_process` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `fk_interface_controller` INT UNSIGNED NOT NULL COMMENT 'Foreign key of interface_controller process that created us' ,
  `fk_fileset` INT UNSIGNED NOT NULL COMMENT 'Foreign key of fileset that\'s being translated.' ,
  `kill_tp` BOOLEAN NOT NULL COMMENT 'Force translor process to stop.' ,
  `started` TIMESTAMP NOT NULL DEFAULT '2000-01-01 00:00:00' COMMENT 'Time this process started.' ,
  `stopped` TIMESTAMP NULL COMMENT 'Time this process stopped' ,
  `filesize_bytes` BIGINT UNSIGNED NULL COMMENT 'Translated file size in bytes' ,
  `tstatus_code` INT NULL COMMENT 'Return status code from translator' ,
  `istatus_code` INT NULL COMMENT 'Status code from iRODS put' ,
  `tru_utime` FLOAT NULL COMMENT 'Translator user mode time' ,
  `tru_stime` FLOAT NULL COMMENT 'Translator system mode time' ,
  `tru_maxrss` INT UNSIGNED NULL COMMENT 'Translator maximum resident set size' ,
  `tru_ixrss` INT UNSIGNED NULL COMMENT 'Translator shared memory size' ,
  `tru_idrss` INT UNSIGNED NULL COMMENT 'Translator unshared memory size' ,
  `tru_isrss` INT UNSIGNED NULL COMMENT 'Translator unshared stack size' ,
  `tru_minflt` INT NULL COMMENT 'Translator page faults not requiring I/O' ,
  `tru_majflt` INT UNSIGNED NULL COMMENT 'Translator page faults requiring I/O' ,
  `tru_nswap` INT UNSIGNED NULL COMMENT 'Translator number of swap outs' ,
  `tru_inblock` INT UNSIGNED NULL COMMENT 'Translator block input operations' ,
  `tru_outblock` INT UNSIGNED NULL COMMENT 'Translator block output operations' ,
  `tru_msgsnd` INT UNSIGNED NULL COMMENT 'Translator messages sent' ,
  `tru_msgrcv` INT UNSIGNED NULL COMMENT 'Translator messages received' ,
  `tru_nsignals` INT UNSIGNED NULL COMMENT 'Translator Signals received' ,
  `tru_nvcsw` INT UNSIGNED NULL COMMENT 'Translator voluntary context switches' ,
  `tru_nivcsw` INT UNSIGNED NULL COMMENT 'Translator involuntary context switches' ,
  `log` TEXT COMMENT 'Log file name.`,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `interface_controller_fk`
    FOREIGN KEY (`fk_interface_controller` )
    REFERENCES `interface_db`.`interface_controller` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fileset_fk_from_translator`
    FOREIGN KEY (`fk_fileset` )
    REFERENCES `interface_db`.`fileset` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'A record of each instance of the translator we create.';

SHOW WARNINGS;
CREATE INDEX `interface_controller_fk` ON `translator_process` (`fk_interface_controller` ASC) ;

SHOW WARNINGS;
CREATE INDEX `fileset_fk_from_translator` ON `translator_process` (`fk_fileset` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `online_status_msg_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `online_status_msg_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `online_status_msg_def` (
  `id` INT UNSIGNED NOT NULL ,
  `name` TINYTEXT NOT NULL COMMENT 'A short name for the message' ,
  `message` TEXT NOT NULL COMMENT 'The message itself.' ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'List of possible online system messages.';

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `status_from_online`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `status_from_online` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `status_from_online` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `fk_online_status_msg` INT UNSIGNED NOT NULL COMMENT 'Foreign key online_status_msg_def online message.' ,
  `experiment` TINYTEXT NULL COMMENT 'Optional short experiment identifier' ,
  `status_text` TEXT NOT NULL COMMENT 'Full text of the message.' ,
  `created` DATETIME NOT NULL COMMENT 'Time created by the online system.' ,
  `offline_ack` DATETIME NULL COMMENT 'Time acknowledged by the offline system.' ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `online_status_fk`
    FOREIGN KEY (`fk_online_status_msg` )
    REFERENCES `interface_db`.`online_status_msg_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Status written by online, read by offline.';

SHOW WARNINGS;
CREATE INDEX `online_status_fk` ON `status_from_online` (`fk_online_status_msg` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `offline_status_msg_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `offline_status_msg_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `offline_status_msg_def` (
  `id` INT UNSIGNED NOT NULL ,
  `name` TINYTEXT NOT NULL COMMENT 'A short name for the message.' ,
  `message` TEXT NOT NULL COMMENT 'The message itself.' ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'List of possible offline status messages';

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `status_from_offline`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `status_from_offline` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `status_from_offline` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `fk_offline_status_msg` INT UNSIGNED NOT NULL COMMENT 'Foreign key of offline status message.' ,
  `experiment` TINYTEXT NULL COMMENT 'Optional experiment related to the status' ,
  `status_text` TEXT NOT NULL COMMENT 'Full text of the status' ,
  `created` DATETIME NOT NULL COMMENT 'Time message written by the offline system.' ,
  `online_ack` DATETIME NULL COMMENT 'Time acknowledged by the online system.' ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `offline_status_fk`
    FOREIGN KEY (`fk_offline_status_msg` )
    REFERENCES `interface_db`.`offline_status_msg_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Status written by offline, read by online.';

SHOW WARNINGS;
CREATE INDEX `offline_status_fk` ON `status_from_offline` (`fk_offline_status_msg` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `archive_interface_controller`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `archive_interface_controller` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `archive_interface_controller` (
  `id` INT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `archive_translator_node`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `archive_translator_node` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `archive_translator_node` (
  `id` INT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `archive_fileset`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `archive_fileset` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `archive_fileset` (
  `id` INT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `archive_files`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `archive_files` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `archive_files` (
  `id` INT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `archive_translator_process`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `archive_translator_process` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `archive_translator_process` (
  `id` INT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `config_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `config_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `config_def` (
  `section` TINYTEXT NOT NULL ,
  `param` TINYTEXT NOT NULL ,
  `value` TINYTEXT NOT NULL ,
  `type` ENUM('Integer','Float','String','Date/Time') NOT NULL ,
  `description` TEXT NULL,
  `instrument` TINYTEXT NULL,
  `experiment` TINYTEXT NULL)
ENGINE = InnoDB
COMMENT = 'Interface DB & controller config params.';

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `cache`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `cache` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `cache` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'This is the uri as seem by the offline side.' ,
  `offline_uri` TEXT NOT NULL COMMENT 'This is the directory URI as seen by the offline side.' ,
  `online_uri` TEXT NOT NULL COMMENT 'Directory URI as seen by the online side' ,
  `avail_bytes` BIGINT UNSIGNED NOT NULL COMMENT 'Bytes available for use as offline->online cache.' ,
  `alloc_bytes` BIGINT UNSIGNED NOT NULL COMMENT 'Bytes allocated by offline cache request for new_fileset.' ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'Directory URIs for cache duty.';

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `files`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `files` ;

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
    REFERENCES `interface_db`.`fileset` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Filenames associated with a fileset.';

SHOW WARNINGS;
CREATE INDEX `fileset_id_fk` ON `files` (`fk_fileset_id` ASC) ;


-- -----------------------------------------------------
-- Table `active_exp`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `active_exp` ;

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
-- Data for table `translator_node`
-- -----------------------------------------------------
SET AUTOCOMMIT=0;
INSERT INTO `translator_node` (`id`, `active`, `node_uri`, `translate_uri`, `log_uri`, `description`) 
  VALUES (0, '1', 'bbt-odf100.slac.stanford.edu', '/data1/rcs/xlate', '/data1/rcs/log', 'Development test machine');
INSERT INTO `translator_node` (`id`, `active`, `node_uri`, `translate_uri`, `log_uri`, `description`) 
  VALUES (0, '1', 'psdss201.pcdsn', 
    '/reg/d/pcds/psdm/hdf5/%(instrument)s/%(experiment)s/%(run_number)06d', 
    '/reg/g/psdm/psdatmgr/ic/log/psdss201', 'Production machine');
INSERT INTO `translator_node` (`id`, `active`, `node_uri`, `translate_uri`, `log_uri`, `description`) 
  VALUES (0, '1', 'psdss202.pcdsn', 
    '/reg/d/pcds/psdm/hdf5/%(instrument)s/%(experiment)s/%(run_number)06d', 
    '/reg/g/psdm/psdatmgr/ic/log/psdss202', 'Production machine');
INSERT INTO `translator_node` (`id`, `active`, `node_uri`, `translate_uri`, `log_uri`, `description`) 
  VALUES (0, '1', 'psdss203.pcdsn', 
    '/reg/d/pcds/psdm/hdf5/%(instrument)s/%(experiment)s/%(run_number)06d', 
    '/reg/g/psdm/psdatmgr/ic/log/psdss203', 'Production machine');

COMMIT;

-- -----------------------------------------------------
-- Data for table `fileset_status_def`
-- do not delete any ID, and do not reuse IDs, any new state
-- should be added at the end with the increasing ID number
-- -----------------------------------------------------
SET AUTOCOMMIT=0;
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (1, 'Initial_Entry', 'Fileset created.');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (2, 'Being_Copied', 'Files in this fileset are being copied to offline.');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (3, 'Waiting_Translation', 'Files in this fileset are waiting to be translated.');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (4, 'Empty_Fileset', 'No files in fileset, or fileset is non-empty when expected empty.');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (5, 'Being_Translated', 'Files in this fileset are being translated.');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (6, 'Translation_Complete', 'Files in this fileset have been translated.');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (7, 'Translation_Error', 'Error status from translator.');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (8, 'Archived_OK', 'The translated files for this fileset has been archived in file manager.');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (9, 'Archive_Error', 'Error status from file manager system.');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (10, 'Complete', 'All processing completed successfully. The translated file has been deleted from the offline cache.');

COMMIT;

-- ---------------------------
-- Data for table "config_def"
-- ---------------------------
INSERT INTO `config_def` (section, param, value, type, description )
  VALUES ('fm-irods', 'filemanager-xtc-dir', '/psdm-zone/exp/%(instrument)s/%(experiment)s/xtc/%(run_number)06d',
          'String', 'Directory for XTC files in file manager' ); 
INSERT INTO `config_def` (section, param, value, type, description )
  VALUES ('fm-irods', 'filemanager-hdf5-dir', '/psdm-zone/exp/%(instrument)s/%(experiment)s/hdf5/%(run_number)06d',
          'String', 'Directory for HDF5 files in file manager' );
INSERT INTO `config_def` (section, param, value, type, description )
  VALUES ('fm-irods', 'filemanager-irods-command', 'ireg', 'String', 'Command to be used for archiving data in iRODS');
INSERT INTO `config_def` (section, param, value, type, description )
  VALUES ('fm-irods', 'irodsHost', 'psdm.slac.stanford.edu', 'String', 'Host name for iRODS server');
INSERT INTO `config_def` (section, param, value, type, description )
  VALUES ('fm-irods', 'irodsPort', '1247', 'Integer', 'iRODS server port number' );
INSERT INTO `config_def` (section, param, value, type, description )
  VALUES ('fm-irods', 'irodsZone', 'psdm-zone', 'String', 'Default zone name for iRODS' );
INSERT INTO `config_def` (section, param, value, type, description )
  VALUES ('fm-irods', 'irodsUserName', 'psdatmgr', 'String', 'iRODS user name' );
INSERT INTO `config_def` (section, param, value, type, description )
  VALUES ('fm-irods', 'irodsAuthFileName', '/u2/ic/.irods/.irodsA', 'String', 'iRODS password file name' );
INSERT INTO `config_def` (section, param, value, type, description )
  VALUES ('fm-irods', 'irodsDefResource', 'testResc', 'String', 'Default iRODS resource name');

INSERT INTO `config_def` (section, param, value, type, description )
  VALUES ('xtc-lustre', 'xtc-location-root', '/reg/d/pcds', 'String', 'Directory with XTC files');
INSERT INTO `config_def` (section, param, value, type, description )
  VALUES ('xtc-lustre', 'xtc-name-pattern', 'i%(instrument)s-e%(experiment)s-r%(run_number)06d-*.xtc', 'String', 'Pattern for names of XTC files');

COMMIT;
