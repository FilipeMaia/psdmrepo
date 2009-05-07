SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';


-- -----------------------------------------------------
-- Table `cache_translator_cpu`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `cache_translator_cpu` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `cache_translator_cpu` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `cpu_uri` VARCHAR(256) NOT NULL COMMENT 'Initially node:network but may be more general later.' ,
  `description` VARCHAR(1024) NOT NULL ,
  `translator_active` BOOLEAN NOT NULL COMMENT 'Set TRUE when Interface Controller starts a translator and set FALSE when it finishes. Insures only one translator active at a time on a given CPU.' ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'Contains one row for each cache/translator CPU.';

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `cache_dir_uri`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `cache_dir_uri` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `cache_dir_uri` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'This is the uri as seem by the offline side.' ,
  `offline_dir_uri` VARCHAR(256) NOT NULL COMMENT 'This is the directory URI as seen by the offline side.' ,
  `avail_bytes` BIGINT UNSIGNED NOT NULL COMMENT 'Bytes available for use as offline->online cache.' ,
  `online_dir_uri` VARCHAR(256) NOT NULL COMMENT 'Directory URI as seen by the online side' ,
  `alloc_bytes` BIGINT UNSIGNED NOT NULL COMMENT 'Bytes allocated by offline cache request for new_fileset.' ,
  `translator_uri` VARCHAR(256) NOT NULL COMMENT 'URI where translator will store the translated file(s).' ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'Directory URIs for cache duty.';

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `instrument_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `instrument_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `instrument_def` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(64) NOT NULL ,
  `description` VARCHAR(1024) NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'Instrumentss supported';

SHOW WARNINGS;
CREATE UNIQUE INDEX `instrument_name_idx` ON `instrument_def` (`name` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `interface_controller`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `interface_controller` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `interface_controller` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `cache_translator_cpu` INT UNSIGNED NOT NULL ,
  `process_id` INT NOT NULL ,
  `started` TIMESTAMP NOT NULL ,
  `stopped` TIMESTAMP NULL ,
  `stop_now` BOOLEAN NOT NULL ,
  `experiment_def` INT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `cache_translator_cpu_from_ic`
    FOREIGN KEY (`cache_translator_cpu` )
    REFERENCES `interface_db`.`cache_translator_cpu` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `experiment_from_ic`
    FOREIGN KEY (`experiment_def` )
    REFERENCES `interface_db`.`instrument_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Instance of an Interface Controller on a cache_trans_cpu.';

SHOW WARNINGS;
CREATE INDEX `cache_translator_cpu_from_ic` ON `interface_controller` (`cache_translator_cpu` ASC) ;

SHOW WARNINGS;
CREATE INDEX `experiment_from_ic` ON `interface_controller` (`experiment_def` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `translator_process`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `translator_process` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `translator_process` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `interface_controller` INT UNSIGNED NOT NULL ,
  `process_id` INT UNSIGNED NOT NULL ,
  `started` TIMESTAMP NOT NULL ,
  `stopped` TIMESTAMP NULL ,
  `total_io` INT UNSIGNED NULL ,
  `total_cpu` INT UNSIGNED NULL ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `interface_controller`
    FOREIGN KEY (`interface_controller` )
    REFERENCES `interface_db`.`interface_controller` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'A record of each instance of the translator we create.';

SHOW WARNINGS;
CREATE INDEX `interface_controller` ON `translator_process` (`interface_controller` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `fileset_status_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `fileset_status_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `fileset_status_def` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(64) NOT NULL ,
  `description` VARCHAR(1024) NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'A list of possible status conditions for a fileset.';

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
  `cache_dir_uri` INT UNSIGNED NOT NULL COMMENT 'Foreign key of directory uri where files of this fileset reside.' ,
  `experiment_def` INT UNSIGNED NOT NULL ,
  `created` TIMESTAMP NOT NULL ,
  `deleted` TIMESTAMP NULL ,
  `fileset_status_def` INT UNSIGNED NOT NULL ,
  `requested_bytes` BIGINT UNSIGNED NOT NULL COMMENT 'Total bytes requested when fileset is created.' ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `fileset_status_from_fileset`
    FOREIGN KEY (`fileset_status_def` )
    REFERENCES `interface_db`.`fileset_status_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `cache_uri_from_fileset`
    FOREIGN KEY (`cache_dir_uri` )
    REFERENCES `interface_db`.`cache_dir_uri` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `experiment_from_fileset`
    FOREIGN KEY (`experiment_def` )
    REFERENCES `interface_db`.`instrument_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'A fileset is the container.';

SHOW WARNINGS;
CREATE INDEX `fileset_status_from_fileset` ON `fileset` (`fileset_status_def` ASC) ;

SHOW WARNINGS;
CREATE INDEX `cache_uri_from_fileset` ON `fileset` (`cache_dir_uri` ASC) ;

SHOW WARNINGS;
CREATE INDEX `experiment_from_fileset` ON `fileset` (`experiment_def` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `filename_status_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `filename_status_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `filename_status_def` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(64) NOT NULL ,
  `description` VARCHAR(1024) NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'A list of possible status conditions for a given filename.';

SHOW WARNINGS;
CREATE UNIQUE INDEX `short_idx` ON `filename_status_def` (`name` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `filetype_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `filetype_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `filetype_def` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(64) NOT NULL ,
  `description` VARCHAR(1024) NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'The file types that offline can send to online.';

SHOW WARNINGS;
CREATE INDEX `filetype_name_idx` ON `filetype_def` (`name` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `filename`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `filename` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `filename` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `filename` VARCHAR(256) NOT NULL ,
  `fileset_id` INT UNSIGNED NOT NULL COMMENT 'Foreign key of the associated fileset.' ,
  `byte_size` BIGINT UNSIGNED NOT NULL ,
  `file_type` INT UNSIGNED NOT NULL COMMENT 'Foreign key id of file type' ,
  `filename_status_def` INT UNSIGNED NOT NULL COMMENT 'Foreign key of file status\n' ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `fileset_id_from_filename`
    FOREIGN KEY (`fileset_id` )
    REFERENCES `interface_db`.`fileset` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `file_status_from_filename`
    FOREIGN KEY (`filename_status_def` )
    REFERENCES `interface_db`.`filename_status_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `file_type_from_filename`
    FOREIGN KEY (`file_type` )
    REFERENCES `interface_db`.`filetype_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Filenames associated with a fileset.';

SHOW WARNINGS;
CREATE INDEX `fileset_id_from_filename` ON `filename` (`fileset_id` ASC) ;

SHOW WARNINGS;
CREATE INDEX `file_status_from_filename` ON `filename` (`filename_status_def` ASC) ;

SHOW WARNINGS;
CREATE INDEX `file_type_from_filename` ON `filename` (`file_type` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `online_status_msg_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `online_status_msg_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `online_status_msg_def` (
  `id` INT UNSIGNED NOT NULL ,
  `name` VARCHAR(64) NOT NULL ,
  `message` VARCHAR(1024) NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'List of possible online system messages.';

SHOW WARNINGS;
CREATE UNIQUE INDEX `offline_status_name_idx` ON `online_status_msg_def` (`name` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `status_from_online`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `status_from_online` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `status_from_online` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `experiment_id` INT UNSIGNED NOT NULL ,
  `online_status_msg_def` INT UNSIGNED NOT NULL ,
  `status_text` VARCHAR(1024) NULL ,
  `offline_ack` BOOLEAN NOT NULL ,
  `time_written` DATETIME NULL ,
  `time_ack` DATETIME NULL ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `experiment_from_onlstatus`
    FOREIGN KEY (`experiment_id` )
    REFERENCES `interface_db`.`instrument_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `online_status_from_onlstatus`
    FOREIGN KEY (`online_status_msg_def` )
    REFERENCES `interface_db`.`online_status_msg_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Status written by online, read by offline.';

SHOW WARNINGS;
CREATE INDEX `experiment_from_onlstatus` ON `status_from_online` (`experiment_id` ASC) ;

SHOW WARNINGS;
CREATE INDEX `online_status_from_onlstatus` ON `status_from_online` (`online_status_msg_def` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `offline_status_msg_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `offline_status_msg_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `offline_status_msg_def` (
  `id` INT UNSIGNED NOT NULL ,
  `name` VARCHAR(64) NOT NULL COMMENT 'A short name for the message.' ,
  `message` VARCHAR(1024) NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'List of possible offline status messages';

SHOW WARNINGS;
CREATE UNIQUE INDEX `offline_status_name_idx` ON `offline_status_msg_def` (`name` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `status_from_offline`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `status_from_offline` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `status_from_offline` (
  `id_status_from_offline` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `experiment_def` INT UNSIGNED NOT NULL ,
  `offline_status_msg_def` INT UNSIGNED NOT NULL ,
  `status_text` VARCHAR(1024) NULL ,
  `online_ack` BOOLEAN NOT NULL ,
  `time_written` DATETIME NOT NULL ,
  `time_ack` DATETIME NULL ,
  PRIMARY KEY (`id_status_from_offline`) ,
  CONSTRAINT `experiment_from_oflstatus`
    FOREIGN KEY (`experiment_def` )
    REFERENCES `interface_db`.`instrument_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `offline_status_from_oflstatus`
    FOREIGN KEY (`offline_status_msg_def` )
    REFERENCES `interface_db`.`offline_status_msg_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Status written by offline, read by online.';

SHOW WARNINGS;
CREATE INDEX `experiment_from_oflstatus` ON `status_from_offline` (`experiment_def` ASC) ;

SHOW WARNINGS;
CREATE INDEX `offline_status_from_oflstatus` ON `status_from_offline` (`offline_status_msg_def` ASC) ;

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
-- Table `archive_cache_translator_cpu`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `archive_cache_translator_cpu` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `archive_cache_translator_cpu` (
  `id` INT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `archive_file_set`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `archive_file_set` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `archive_file_set` (
  `id` INT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `archive_file_name`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `archive_file_name` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `archive_file_name` (
  `id` INT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `archive_translator`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `archive_translator` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `archive_translator` (
  `id` INT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `experiment_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `experiment_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `experiment_def` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(64) NOT NULL ,
  `description` VARCHAR(1024) NOT NULL ,
  `instrument_id` INT UNSIGNED NOT NULL COMMENT 'Foreign key referring to the instrument used.' ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `instrument_from_def`
    FOREIGN KEY (`instrument_id` )
    REFERENCES `interface_db`.`instrument_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Experiments that use instruments.';

SHOW WARNINGS;
CREATE INDEX `instrument_from_def` ON `experiment_def` (`instrument_id` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `config_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `config_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `config_def` (
  `section` VARCHAR(64) NOT NULL ,
  `param` VARCHAR(64) NOT NULL ,
  `value` VARCHAR(256) NOT NULL ,
  `type` ENUM('Integer','Float','String','Date/Time') NOT NULL ,
  `description` VARCHAR(1024) NULL ,
  PRIMARY KEY (`section`, `param`) )
ENGINE = InnoDB
COMMENT = 'Interface DB & controller config params.';

SHOW WARNINGS;
CREATE INDEX `section_idx` ON `config_def` (`section` ASC, `param` ASC) ;

SHOW WARNINGS;
CREATE INDEX `param_idx` ON `config_def` (`param` ASC) ;

SHOW WARNINGS;

DELIMITER //
DROP procedure IF EXISTS `new_file_set` //
SHOW WARNINGS//
CREATE PROCEDURE `interface_db`.`new_file_set` (
                IN in_experiment_id INT UNSIGNED, 
                IN in_requested_bytes BIGINT UNSIGNED, 
                OUT out_fileset_id INT UNSIGNED,
                OUT out_dir_uri VARCHAR(256))
BEGIN
  DECLARE cache_avail TINYINT DEFAULT 1;   /* Assume cache available */
  DECLARE id_uri INT UNSIGNED;
  DECLARE offline_uri VARCHAR(256);
  /*
  ** Find a cache uri that has enough space and whose CPU is not translating.
  */
  DECLARE uri_cur CURSOR FOR SELECT id, online_dir_uri FROM cache_dir_uri AS uri
  JOIN cache_translator_cpu
    ON (uri.avail_bytes >= requested_bytes AND
        cache_translator_cpu.translator_active != TRUE AND
        uri.cache_translator_cpu = cache_translator_cpu.id);
 
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET cache_avail = 0;
   
  OPEN uri_cur;
  FETCH uri_cur INTO id_uri, offline_uri;

  IF cache_avail THEN
    /*
    ** Update available bytes in the cache_uri table and create a new fileset.
    */
    UPDATE cache_dir_uri SET avail_bytes = avail_bytes - requested_bytes
    WHERE cache_dir_uri.id = id_uri;

    INSERT INTO fileset
    SET cache_dir_uri = id_uri, experiment_def = in_experiment_id, 
        created   = NOW(), deleted = NULL, 
        fileset_status_def = func_get_id ('fileset_status', 'Initial_Entry'),
        requested_bytes = in_requested_bytes;
   
    SET out_fileset_id = LAST_INSERT_ID();
    SET out_dir_uri = offline_uri;
  ELSE
/*
** Return id_fileset = 0 if no cache available
*/    SET out_fileset_id = 0;
  END IF;
END//
SHOW WARNINGS//
DROP procedure IF EXISTS `add_file_to_set` //
SHOW WARNINGS//
/*
** Add filename to existing fileset. Out_status returns 0 if not success.
*/
CREATE PROCEDURE `interface_db`.`add_file_to_set` (IN in_fileset INT UNSIGNED, 
                IN in_filename VARCHAR(255), IN in_filesize_bytes BIGINT UNSIGNED,
                IN in_filetype INT UNSIGNED, OUT out_status TINYINT)
BEGIN
  /*
  ** Find the fileset.
  */
  DECLARE fileset_exists TINYINT DEFAULT -1;        /* Assume fileset exists */ 
  DECLARE lfileset_id INT UNSIGNED DEFAULT 0; /* Local fileset id */
  DECLARE fs_cur CURSOR FOR SELECT id FROM fileset WHERE id = in_fileset;
  DECLARE CONTINUE HANDLER FOR NOT FOUND SET fileset_exists = 0;

  SET out_status = -1;  /* Assume we'll succeed */
  
  OPEN fs_cur;
  IF fileset_exists THEN
    /*
    ** Fetch id_fileset and insert new filename row.
    */
    FETCH fs_cur INTO lfileset_id;
    
    INSERT INTO filename
    SET fileset_id = lfileset_id, filename = in_filename, 
        byte_size = in_filesize_bytes, file_type = in_filetype,
        file_status = func_get_id ('filename_status', 'Initial_Entry');
   
  ELSE
    /*
    ** Return out_status = 0 if fileset doesn't exist.
    */
    SET out_status = 0;
  END IF;  
END//
SHOW WARNINGS//
DROP function IF EXISTS `func_get_id` //
SHOW WARNINGS//
CREATE FUNCTION `interface_db`.`func_get_id` (
                 in_table_name VARCHAR(256),
                 in_name VARCHAR(64)) RETURNS INT UNSIGNED
BEGIN
/*
** General function to return the primary key id from the following tables:
** experiment_id, filename_status_def, fileset_status_def, filetype_def,
** offline_status_msg_def, online_status_msg_def.
** All have a name varchar(64) name key.
*/
  DECLARE lid INT UNSIGNED;

  SELECT id FROM in_table_name AS tn 
           WHERE tn.name = in_name INTO lid;

  RETURN lid;
END//
SHOW WARNINGS//
DROP procedure IF EXISTS `get_fileset_with_status` //
SHOW WARNINGS//
CREATE PROCEDURE `interface_db`.`get_fileset_with_status` (
                 IN in_status INT UNSIGNED,
                 IN in_experiment INT UNSIGNED,
                 OUT out_fileset_id INT UNSIGNED)
BEGIN
/*
** Find a fileset for the experiment with this status.
*/
  SELECT id FROM fileset AS fs
     WHERE fs.experiment_def = in_experiment AND fs.fileset_status_def = in_status
     INTO out_fileset_id;
END//
SHOW WARNINGS//
DROP procedure IF EXISTS `change_fileset_status` //
SHOW WARNINGS//
CREATE PROCEDURE `interface_db`.`change_fileset_status` (
                 IN in_fileset_id INT UNSIGNED,
                 IN in_status TINYINT UNSIGNED)
                 
BEGIN
     UPDATE fileset SET fileset_status_def = in_status 
         WHERE fileset.id = in_fileset_id;
END//
SHOW WARNINGS//
DROP procedure IF EXISTS `change_filename_status` //
SHOW WARNINGS//
CREATE PROCEDURE `interface_db`.`change_filename_status` (
                 IN in_filename_id INT UNSIGNED,
                 IN in_status TINYINT UNSIGNED)
BEGIN
     UPDATE filename SET filename_status_def = in_status
         WHERE filename.id = in_filename_id;

END//
SHOW WARNINGS//

DELIMITER ;

-- -----------------------------------------------------
-- Data for table `cache_translator_cpu`
-- -----------------------------------------------------
SET AUTOCOMMIT=0;
INSERT INTO `cache_translator_cpu` (`id`, `cpu_uri`, `description`, `translator_active`) VALUES (0, 'bbt-odf100.slac.stanford.edu', 'Development test machine', false);

COMMIT;

-- -----------------------------------------------------
-- Data for table `instrument_def`
-- -----------------------------------------------------
SET AUTOCOMMIT=0;
INSERT INTO `instrument_def` (`id`, `name`, `description`) VALUES (0, 'AMOS', 'Atomic Molecular & Optical Science');
INSERT INTO `instrument_def` (`id`, `name`, `description`) VALUES (0, 'XPP', 'X-Ray Pump Probe');
INSERT INTO `instrument_def` (`id`, `name`, `description`) VALUES (0, 'CXI', 'Coherent X-Ray Imaging');
INSERT INTO `instrument_def` (`id`, `name`, `description`) VALUES (0, 'XCS', 'X-Ray Correlation Spectroscopy');

COMMIT;

-- -----------------------------------------------------
-- Data for table `fileset_status_def`
-- -----------------------------------------------------
SET AUTOCOMMIT=0;
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (0, 'Initial_Entry', 'Fileset created');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (0, 'Being_Copied', 'Files in this fileset are being copied to offline');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (0, 'Waiting_Translation', 'Files in this fileset are waiting translation');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (0, 'Being_Translated', 'Files in this fileset are being translated');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (0, 'Translation_Complete', 'Files in this fileset have been translated');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (0, 'Translation_Failed', 'Translation of this fileset has failed');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (0, 'Entered_into_iRODS', 'The translated file for this fileset has been entered into the iRODS file manager');
INSERT INTO `fileset_status_def` (`id`, `name`, `description`) VALUES (0, 'Deleted_fromCache', 'The fileset and its files have been deleted from the offline cache');

COMMIT;

-- -----------------------------------------------------
-- Data for table `filename_status_def`
-- -----------------------------------------------------
SET AUTOCOMMIT=0;
INSERT INTO `filename_status_def` (`id`, `name`, `description`) VALUES (0, 'Initial_Entry', 'Status when file is initially entered into a fileset');
INSERT INTO `filename_status_def` (`id`, `name`, `description`) VALUES (0, 'Being_Copied', 'File is being copied from online to offline');
INSERT INTO `filename_status_def` (`id`, `name`, `description`) VALUES (0, 'Waiting_Translation', 'File is waiting to be translated');
INSERT INTO `filename_status_def` (`id`, `name`, `description`) VALUES (0, 'Being_Translated', 'Translator is working on this file');
INSERT INTO `filename_status_def` (`id`, `name`, `description`) VALUES (0, 'Translation_Complete', 'Successful translation completed');
INSERT INTO `filename_status_def` (`id`, `name`, `description`) VALUES (0, 'Translation_Failed', 'Error during translation');
INSERT INTO `filename_status_def` (`id`, `name`, `description`) VALUES (0, 'Entered_into_iRODS', 'The translated file that contains this file has been entered into the iRODS file manafer');
INSERT INTO `filename_status_def` (`id`, `name`, `description`) VALUES (0, 'Deleted_from_Cache', 'File has been deleted from the cache area');

COMMIT;

-- -----------------------------------------------------
-- Data for table `filetype_def`
-- -----------------------------------------------------
SET AUTOCOMMIT=0;
INSERT INTO `filetype_def` (`id`, `name`, `description`) VALUES (0, 'XTC', 'eXtended Tagged Container');
INSERT INTO `filetype_def` (`id`, `name`, `description`) VALUES (0, 'EPICS', 'Slow EPICS data');

COMMIT;

-- -----------------------------------------------------
-- Data for table `experiment_def`
-- -----------------------------------------------------
SET AUTOCOMMIT=0;
INSERT INTO `experiment_def` (`id`, `name`, `description`, `instrument_id`) VALUES (0, 'First_AMOS', 'Sample experiment', 0);

COMMIT;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
