SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';


-- -----------------------------------------------------
-- Table `translator_cpu`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `translator_cpu` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `translator_cpu` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `cpu_uri` VARCHAR(255) NOT NULL COMMENT 'Initially node:network but may be more general later.' ,
  `description` VARCHAR(1023) NOT NULL ,
  `translator_active` BOOLEAN NOT NULL COMMENT 'Set TRUE when Interface Controller starts a translator and set FALSE when it finishes. Insures only one translator active at a time on a given CPU.' ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'Contains one row for each cache/translator CPU.';

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `cache`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `cache` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `cache` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT COMMENT 'This is the uri as seem by the offline side.' ,
  `offline_uri` VARCHAR(256) NOT NULL COMMENT 'This is the directory URI as seen by the offline side.' ,
  `avail_bytes` BIGINT UNSIGNED NOT NULL COMMENT 'Bytes available for use as offline->online cache.' ,
  `online_uri` VARCHAR(256) NOT NULL COMMENT 'Directory URI as seen by the online side' ,
  `alloc_bytes` BIGINT UNSIGNED NOT NULL COMMENT 'Bytes allocated by offline cache request for new_fileset.' ,
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
  `name` VARCHAR(63) NOT NULL ,
  `description` VARCHAR(1023) NOT NULL ,
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
    REFERENCES `interface_db`.`translator_cpu` (`id` )
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
  `fk_interface_controller` INT UNSIGNED NOT NULL ,
  `process_id` INT UNSIGNED NOT NULL ,
  `started` TIMESTAMP NOT NULL ,
  `stopped` TIMESTAMP NULL ,
  `total_io` INT UNSIGNED NULL ,
  `total_cpu` INT UNSIGNED NULL ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `interface_controller`
    FOREIGN KEY (`fk_interface_controller` )
    REFERENCES `interface_db`.`interface_controller` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'A record of each instance of the translator we create.';

SHOW WARNINGS;
CREATE INDEX `interface_controller` ON `translator_process` (`fk_interface_controller` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `fileset_status_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `fileset_status_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `fileset_status_def` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(63) NOT NULL ,
  `description` VARCHAR(1023) NOT NULL ,
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
  `fk_cache` INT UNSIGNED NOT NULL COMMENT 'Foreign key of directory uri where files of this fileset reside.' ,
  `fk_fileset_status` INT UNSIGNED NOT NULL ,
  `fk_experiment` INT UNSIGNED NOT NULL ,
  `requested_bytes` BIGINT UNSIGNED NOT NULL COMMENT 'Total bytes requested when fileset is created.' ,
  `used_bytes` BIGINT UNSIGNED NOT NULL ,
  `created` TIMESTAMP NOT NULL ,
  `deleted` TIMESTAMP NULL ,
  `locked` BOOLEAN NOT NULL ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `fileset_status_fk`
    FOREIGN KEY (`fk_fileset_status` )
    REFERENCES `interface_db`.`fileset_status_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `cache_fk`
    FOREIGN KEY (`fk_cache` )
    REFERENCES `interface_db`.`cache` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `experiment_fk`
    FOREIGN KEY (`fk_experiment` )
    REFERENCES `interface_db`.`instrument_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'A fileset is the container.';

SHOW WARNINGS;
CREATE INDEX `fileset_status_fk` ON `fileset` (`fk_fileset_status` ASC) ;

SHOW WARNINGS;
CREATE INDEX `cache_fk` ON `fileset` (`fk_cache` ASC) ;

SHOW WARNINGS;
CREATE INDEX `experiment_fk` ON `fileset` (`fk_experiment` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `filetype_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `filetype_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `filetype_def` (
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(63) NOT NULL ,
  `description` VARCHAR(1023) NULL ,
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
  `fk_fileset_id` INT UNSIGNED NOT NULL COMMENT 'Foreign key of the associated fileset.' ,
  `fk_fileset_status` INT UNSIGNED NOT NULL COMMENT 'Foreign key of file status\n' ,
  `fk_file_type` INT UNSIGNED NOT NULL COMMENT 'Foreign key id of file type' ,
  `filename` VARCHAR(256) NOT NULL ,
  `byte_size` BIGINT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `fileset_id_fk`
    FOREIGN KEY (`fk_fileset_id` )
    REFERENCES `interface_db`.`fileset` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `file_type_fk`
    FOREIGN KEY (`fk_file_type` )
    REFERENCES `interface_db`.`filetype_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `fileset_status_fkn`
    FOREIGN KEY (`fk_fileset_status` )
    REFERENCES `interface_db`.`fileset_status_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Filenames associated with a fileset.';

SHOW WARNINGS;
CREATE INDEX `fileset_id_fk` ON `filename` (`fk_fileset_id` ASC) ;

SHOW WARNINGS;
CREATE INDEX `file_type_fk` ON `filename` (`fk_file_type` ASC) ;

SHOW WARNINGS;
CREATE INDEX `fileset_status_fkn` ON `filename` (`fk_fileset_status` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `online_status_msg_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `online_status_msg_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `online_status_msg_def` (
  `id` INT UNSIGNED NOT NULL ,
  `name` VARCHAR(63) NOT NULL ,
  `message` VARCHAR(1023) NULL ,
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
  `fk_experiment` INT UNSIGNED NOT NULL ,
  `fk_online_status_msg` INT UNSIGNED NOT NULL ,
  `status_text` VARCHAR(1023) NULL ,
  `offline_ack` BOOLEAN NOT NULL ,
  `time_written` DATETIME NULL ,
  `time_ack` DATETIME NULL ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `experiment_from_online_fk`
    FOREIGN KEY (`fk_experiment` )
    REFERENCES `interface_db`.`instrument_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `online_status_fk`
    FOREIGN KEY (`fk_online_status_msg` )
    REFERENCES `interface_db`.`online_status_msg_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Status written by online, read by offline.';

SHOW WARNINGS;
CREATE INDEX `experiment_from_online_fk` ON `status_from_online` (`fk_experiment` ASC) ;

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
  `name` VARCHAR(63) NOT NULL COMMENT 'A short name for the message.' ,
  `message` VARCHAR(1023) NULL ,
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
  `id` INT UNSIGNED NOT NULL AUTO_INCREMENT ,
  `fk_experiment` INT UNSIGNED NOT NULL ,
  `fk_offline_status_msg` INT UNSIGNED NOT NULL ,
  `status_text` VARCHAR(1023) NULL ,
  `online_ack` BOOLEAN NOT NULL ,
  `time_written` DATETIME NOT NULL ,
  `time_ack` DATETIME NULL ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `experiment_from_offline_fk`
    FOREIGN KEY (`fk_experiment` )
    REFERENCES `interface_db`.`instrument_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `offline_status_fk`
    FOREIGN KEY (`fk_offline_status_msg` )
    REFERENCES `interface_db`.`offline_status_msg_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Status written by offline, read by online.';

SHOW WARNINGS;
CREATE INDEX `experiment_from_offline_fk` ON `status_from_offline` (`fk_experiment` ASC) ;

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
-- Table `archive_translator_cpu`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `archive_translator_cpu` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `archive_translator_cpu` (
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
  `fk_instrument` INT UNSIGNED NOT NULL COMMENT 'Foreign key referring to the instrument used.' ,
  `name` VARCHAR(63) NOT NULL ,
  `description` VARCHAR(1023) NOT NULL ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `instrument_fk`
    FOREIGN KEY (`fk_instrument` )
    REFERENCES `interface_db`.`instrument_def` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Experiments that use instruments.';

SHOW WARNINGS;
CREATE INDEX `instrument_fk` ON `experiment_def` (`fk_instrument` ASC) ;

SHOW WARNINGS;

-- -----------------------------------------------------
-- Table `config_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `config_def` ;

SHOW WARNINGS;
CREATE  TABLE IF NOT EXISTS `config_def` (
  `section` VARCHAR(63) NOT NULL ,
  `param` VARCHAR(63) NOT NULL ,
  `value` VARCHAR(255) NOT NULL ,
  `type` ENUM('Integer','Float','String','Date/Time') NOT NULL ,
  `description` VARCHAR(1023) NULL ,
  PRIMARY KEY (`section`, `param`) )
ENGINE = InnoDB
COMMENT = 'Interface DB & controller config params.';

SHOW WARNINGS;
CREATE INDEX `section_idx` ON `config_def` (`section` ASC, `param` ASC) ;

SHOW WARNINGS;
CREATE INDEX `param_idx` ON `config_def` (`param` ASC) ;

SHOW WARNINGS;

DELIMITER //
DROP procedure IF EXISTS `add_file_to_set` //
SHOW WARNINGS//
/*
** Add filename to existing fileset. Out_status returns NULL if not success.
*/
CREATE PROCEDURE `interface_db`.`add_file_to_set` (
             IN in_fileset INT UNSIGNED, 
             IN in_filetype INT UNSIGNED,
             IN in_filename VARCHAR(255), 
             IN in_filesize_bytes BIGINT UNSIGNED,
             OUT out_status TINYINT)
BEGIN
     DECLARE fileset_exists TINYINT DEFAULT 1;      /* Assume fileset exists */ 
     DECLARE lcl_fileset_id INT UNSIGNED DEFAULT 0; /* Local fileset id */
     DECLARE lcl_filename_status_id INT UNSIGNED;
     DECLARE lcl_used_bytes BIGINT UNSIGNED;
     DECLARE lcl_requested_bytes BIGINT UNSIGNED;
     DECLARE rows INT;
     DECLARE fs_cur CURSOR FOR SELECT id FROM fileset WHERE id = in_fileset;

     DECLARE CONTINUE HANDLER FOR NOT FOUND SET fileset_exists = 0;

     SET out_status = NULL;  /* Assume failure */
  
     OPEN fs_cur;
     FETCH fs_cur INTO lcl_fileset_id;
     /* Check if filetype exists */
     SELECT COUNT(*) FROM filetype_def WHERE id = in_filetype INTO rows;
     SELECT used_bytes, requested_bytes FROM fileset AS fs WHERE fs.id = lcl_fileset_id
         INTO lcl_used_bytes, lcl_requested_bytes;
     IF fileset_exists AND 
         lcl_used_bytes + in_filesize_bytes <= lcl_requested_bytes
         AND rows = 1 THEN 
         /*
         ** Insert new filename row and update used_bytes in fileset.
         */
         SELECT id FROM fileset_status_def AS fsd
             WHERE fsd.name = 'Initial_Entry' INTO lcl_filename_status_id;
         INSERT INTO filename
         SET fk_fileset_id = lcl_fileset_id, 
             fk_fileset_status = lcl_filename_status_id,
             fk_file_type = in_filetype, filename = in_filename, 
             byte_size = in_filesize_bytes;
         UPDATE fileset SET used_bytes = used_bytes + in_filesize_bytes
             WHERE lcl_fileset_id = fileset.id;
         SET out_status = 1;   /* Set OK status */
     END IF;
END//
SHOW WARNINGS//
DROP procedure IF EXISTS `get_fileset_with_status` //
SHOW WARNINGS//
/*
** Find the oldest fileset with the specified status that is not locked.
** This routine locks the fileset so only one user can get it. Any change
** in the fileset's status unlocks it.
*/
CREATE PROCEDURE `interface_db`.`get_fileset_with_status` (
                 IN in_status INT UNSIGNED,
                 IN in_experiment INT UNSIGNED,
                 OUT out_fileset_id INT UNSIGNED)
BEGIN
     DECLARE fs_avail TINYINT;
     DECLARE fs_cur CURSOR FOR SELECT id FROM fileset AS fs
         WHERE fs.fk_experiment = in_experiment AND fs.fk_fileset_status = in_status 
         AND fs.locked = FALSE ORDER BY fs.created ASC;
 
     DECLARE CONTINUE HANDLER FOR NOT FOUND SET fs_avail = 0;
     /*
     ** Find a fileset for the experiment with this status.
     */
     SET fs_avail = 1;           /* Assume a match */
     OPEN fs_cur;
     FETCH fs_cur INTO out_fileset_id;   /* Get oldest fileset that matches */ 
     IF (fs_avail = 1) THEN
     
         UPDATE fileset SET locked = TRUE
             WHERE fileset.id = out_fileset_id;
     ELSE
         SET out_fileset_id = NULL;
     END IF;
END//
SHOW WARNINGS//
DROP procedure IF EXISTS `change_fileset_status` //
SHOW WARNINGS//
/*
** Update the fileset status and the status of all files it contains
*/
CREATE PROCEDURE `interface_db`.`change_fileset_status` (
                 IN in_fileset_id INT UNSIGNED,
                 IN in_status TINYINT UNSIGNED,
                 OUT out_status TINYINT)
BEGIN
     DECLARE fsn_exists TINYINT DEFAULT 1;
     DECLARE lcl_id INT UNSIGNED;
     DECLARE fs_cur CURSOR FOR SELECT id FROM fileset AS fs
     WHERE fs.id = in_fileset_id;
     DECLARE fn_cur CURSOR FOR SELECT id FROM filename AS fn
     WHERE fn.fk_fileset_id = in_fileset_id;

     DECLARE CONTINUE HANDLER FOR NOT FOUND SET fsn_exists = 0;

     SET out_status = 1;   /* Assume all will be well */
     OPEN fs_cur;
     FETCH fs_cur INTO lcl_id;
     IF (fsn_exists = 1) THEN
         UPDATE fileset SET fk_fileset_status = in_status, locked = FALSE 
            WHERE fileset.id = in_fileset_id;
         /*
         ** Update all filenames in the set with this status.
         */
         OPEN fn_cur;
         fnl: LOOP
             FETCH fn_cur INTO lcl_id;
             IF fsn_exists = 0 THEN LEAVE fnl; END IF;
             UPDATE filename SET fk_fileset_status = in_status 
             WHERE filename.id = lcl_id; 
         END LOOP fnl;
     ELSE    /* No such fileset */
         SET out_status = NULL;
     END IF;
END//
SHOW WARNINGS//
DROP procedure IF EXISTS `change_filename_status` //
SHOW WARNINGS//
/*
** Set filename to specified status
*/
CREATE PROCEDURE `interface_db`.`change_filename_status` (
                 IN in_filename_id INT UNSIGNED,
                 IN in_status INT UNSIGNED,
                 OUT out_status TINYINT)
BEGIN
     DECLARE fn_exists TINYINT DEFAULT 1;
     DECLARE lcl_id INT UNSIGNED;
     DECLARE fn_cur CURSOR FOR SELECT id FROM filename AS fn
         WHERE fn.id = in_filename_id;

     DECLARE CONTINUE HANDLER FOR NOT FOUND SET fn_exists = 0;

     SET out_status = 1;   /* Assume success */
     OPEN fn_cur;
     FETCH fn_cur INTO lcl_id;
     IF (fn_exists = 1) THEN
         UPDATE filename SET fk_fileset_status = in_status 
            WHERE filename.id = in_filename_id;
     ELSE    /* No such filename id */
         SET out_status = NULL;
     END IF;
END//
SHOW WARNINGS//
DROP procedure IF EXISTS `new_fileset` //
SHOW WARNINGS//
/*
** Create a new fileset if we can find cache space for it.
*/
CREATE PROCEDURE `interface_db`.`new_fileset` (
                IN in_experiment_id INT UNSIGNED, 
                IN in_requested_bytes BIGINT UNSIGNED, 
                OUT out_fileset_id INT UNSIGNED,
                OUT out_online_uri VARCHAR(255))

BEGIN
     DECLARE cache_avail TINYINT;
     DECLARE lcl_id INT UNSIGNED DEFAULT 0;
     DECLARE lcl_online_uri VARCHAR(255);
     DECLARE lcl_fileset_status INT UNSIGNED;
     /*
     ** Find a cache uri that has enough space. First try to find one
     ** that has nothing allocated. If all cache directories have some 
     ** space allocated then find the one with the maximum available space.
     */
     DECLARE cache0_cur CURSOR FOR SELECT id, online_uri FROM cache
         WHERE (cache.alloc_bytes = 0) AND 
         (cache.avail_bytes >= in_requested_bytes);

     DECLARE cachemax_cur CURSOR for SELECT id, online_uri FROM cache
         WHERE  cache.avail_bytes >= in_requested_bytes
             ORDER BY cache.avail_bytes DESC;
 
     DECLARE CONTINUE HANDLER FOR NOT FOUND SET cache_avail = 0;

     SET cache_avail = 1;
     OPEN cache0_cur;
     FETCH cache0_cur INTO lcl_id, lcl_online_uri;
     IF (cache_avail = 0) THEN
         SET cache_avail = 1;
         OPEN cachemax_cur;
         FETCH cachemax_cur INTO lcl_id, lcl_online_uri;
     END IF;
    
     IF (cache_avail = 1) THEN
         /*
         ** Update available bytes in the cache table and create a new fileset.
         */
         UPDATE cache SET avail_bytes = avail_bytes - in_requested_bytes,
                          alloc_bytes = alloc_bytes + in_requested_bytes
             WHERE cache.id = lcl_id;
         /*
         ** Get id of "Initial_Entry" for fileset_status
         */
         SELECT id FROM fileset_status_def AS fsd 
             WHERE fsd.name = 'Initial_Entry' INTO lcl_fileset_status;
         /*
         ** Create new fileset row
         */
         INSERT INTO fileset
         SET 
         fk_cache = lcl_id, 
         fk_fileset_status = lcl_fileset_status, 
         fk_experiment = in_experiment_id, 
         requested_bytes = in_requested_bytes,
         used_bytes = 0,
         created   = NOW(), deleted = NULL, locked = FALSE; 
          
         SET out_fileset_id = LAST_INSERT_ID();
         SET out_online_uri = lcl_online_uri;
     ELSE
         /*
         ** Return id_fileset = uri = NULL if no cache available
         */
         SET out_fileset_id = NULL;
         SET out_online_uri = NULL;
     END IF;
END//
SHOW WARNINGS//
DROP procedure IF EXISTS `get_def_id` //
SHOW WARNINGS//
/*
** Implement later when dynamic SQL works.
*/
CREATE PROCEDURE `interface_db`.`get_def_id` (
                 IN in_table VARCHAR(63), IN in_name VARCHAR(63),
                 OUT out_id INT UNSIGNED)
BEGIN
  SET @int = in_table;
  SET @inn = in_name;
  SET @oid = out_id;
  SET @s1 = CONCAT('SELECT id FROM ', in_table, ' AS it WHERE it.name = ',
                    @inn, ' INTO ', @oid);
  SELECT @s1;
  SET @s = 'SELECT id FROM \'?\' AS it WHERE it.name = \'?\' INTO ?';
  SELECT @s;
  PREPARE stmt FROM @s;
  SELECT @s;
  EXECUTE stmti USING @int, @inn, @oid;
  DEALLOCATE PREPARE stmt;
END//
SHOW WARNINGS//

DELIMITER ;

-- -----------------------------------------------------
-- Data for table `translator_cpu`
-- -----------------------------------------------------
SET AUTOCOMMIT=0;
INSERT INTO `translator_cpu` (`id`, `cpu_uri`, `description`, `translator_active`) VALUES (0, 'bbt-odf100.slac.stanford.edu', 'Development test machine', false);

COMMIT;

-- -----------------------------------------------------
-- Data for table `cache`
-- -----------------------------------------------------
SET AUTOCOMMIT=0;
INSERT INTO `cache` (`id`, `offline_uri`, `avail_bytes`, `online_uri`, `alloc_bytes`) VALUES (0, 'bbt-odf100.slac.stanford.edu/data', 2000000, 'bbt-odf100.slac.stanford.edu/data', 0);
INSERT INTO `cache` (`id`, `offline_uri`, `avail_bytes`, `online_uri`, `alloc_bytes`) VALUES (0, 'bbt-odf100.slac.stanford.edu/data1', 3000000, 'bbt-odf100.slac.stanford.edu/data1', 0);

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
INSERT INTO `experiment_def` (`id`, `fk_instrument`, `name`, `description`) VALUES (0, 1, 'First_AMOS', 'Sample experiment');

COMMIT;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
