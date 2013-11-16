SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `LOGBOOK` ;
USE `LOGBOOK`;

-- -----------------------------------------------------
-- Table `LOGBOOK`.`RUN`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`RUN` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`RUN` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `num` INT NOT NULL ,
  `exper_id` INT NOT NULL ,
  `type` ENUM('DATA','CALIB','TEST') NOT NULL DEFAULT 'DATA' ,
  `begin_time` BIGINT UNSIGNED NOT NULL ,
  `end_time` BIGINT UNSIGNED NULL ,
  UNIQUE INDEX `RUN_IDX_1` (`exper_id` ASC, `num` ASC) ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`RUN_PARAM`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`RUN_PARAM` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`RUN_PARAM` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `param` VARCHAR(255) NOT NULL ,
  `exper_id` INT NOT NULL ,
  `type` ENUM('INT','DOUBLE','TEXT') NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE INDEX `RUN_PARAM_IDX_1` (`param` ASC, `exper_id` ASC) )
ENGINE = InnoDB
COMMENT = 'Key names and types for variable parameters of runs';


-- -----------------------------------------------------
-- Table `LOGBOOK`.`RUN_VAL_INT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`RUN_VAL_INT` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`RUN_VAL_INT` (
  `run_id` INT NOT NULL AUTO_INCREMENT ,
  `param_id` INT NOT NULL ,
  `val` INT NULL ,
  PRIMARY KEY (`run_id`, `param_id`) ,
  INDEX `RUN_VAL_INT_FK_1` (`run_id` ASC) ,
  INDEX `RUN_VAL_INT_FK_2` (`param_id` ASC) ,
  CONSTRAINT `RUN_VAL_INT_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `LOGBOOK`.`RUN` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `RUN_VAL_INT_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `LOGBOOK`.`RUN_PARAM` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`RUN_VAL_TEXT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`RUN_VAL_TEXT` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`RUN_VAL_TEXT` (
  `run_id` INT NOT NULL AUTO_INCREMENT ,
  `param_id` INT NOT NULL ,
  `val` MEDIUMTEXT NULL ,
  INDEX `RUN_VAL_TEXT_FK_1` (`run_id` ASC) ,
  INDEX `RUN_VAL_TEXT_FK_2` (`param_id` ASC) ,
  PRIMARY KEY (`run_id`, `param_id`) ,
  CONSTRAINT `RUN_VAL_TEXT_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `LOGBOOK`.`RUN` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `RUN_VAL_TEXT_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `LOGBOOK`.`RUN_PARAM` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`RUN_VAL_DOUBLE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`RUN_VAL_DOUBLE` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`RUN_VAL_DOUBLE` (
  `run_id` INT NOT NULL AUTO_INCREMENT ,
  `param_id` INT NOT NULL ,
  `val` DOUBLE NULL ,
  INDEX `RUN_VAL_DOUBLE_FK_1` (`run_id` ASC) ,
  INDEX `RUN_VAL_DOUBLE_FK_2` (`param_id` ASC) ,
  PRIMARY KEY (`run_id`, `param_id`) ,
  CONSTRAINT `RUN_VAL_DOUBLE_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `LOGBOOK`.`RUN` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `RUN_VAL_DOUBLE_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `LOGBOOK`.`RUN_PARAM` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`RUN_VAL`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`RUN_VAL` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`RUN_VAL` (
  `run_id` INT NOT NULL ,
  `param_id` INT NOT NULL ,
  `source` VARCHAR(255) NOT NULL ,
  `updated` BIGINT UNSIGNED NOT NULL ,
  INDEX `RUN_VAL_FK_1` (`run_id` ASC) ,
  INDEX `RUN_VAL_FK_2` (`param_id` ASC) ,
  PRIMARY KEY (`param_id`, `run_id`) ,
  CONSTRAINT `RUN_VAL_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `LOGBOOK`.`RUN` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `RUN_VAL_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `LOGBOOK`.`RUN_PARAM` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`SHIFT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`SHIFT` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`SHIFT` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `exper_id` INT NOT NULL ,
  `begin_time` BIGINT UNSIGNED NOT NULL ,
  `end_time` BIGINT UNSIGNED NULL ,
  `leader` VARCHAR(32) NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`HEADER`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`HEADER` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`HEADER` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `exper_id` INT NOT NULL ,
  `shift_id` INT NULL ,
  `run_id` INT NULL ,
  `relevance_time` BIGINT UNSIGNED NULL ,
  PRIMARY KEY (`id`) ,
  INDEX `HEADER_FK_1` (`shift_id` ASC) ,
  INDEX `HEADER_FK_2` (`run_id` ASC) ,
  CONSTRAINT `HEADER_FK_1`
    FOREIGN KEY (`shift_id` )
    REFERENCES `LOGBOOK`.`SHIFT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `HEADER_FK_2`
    FOREIGN KEY (`run_id` )
    REFERENCES `LOGBOOK`.`RUN` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`ENTRY`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`ENTRY` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`ENTRY` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `hdr_id` INT NOT NULL ,
  `parent_entry_id` INT NULL ,
  `insert_time` BIGINT UNSIGNED NOT NULL ,
  `author` VARCHAR(32) NOT NULL ,
  `content` MEDIUMTEXT NOT NULL ,
  `content_type` ENUM('TEXT','HTML') NOT NULL ,
  `deleted_time` BIGINT UNSIGNED DEFAULT NULL ,
  `deleted_by` VARCHAR(32) DEFAULT NULL ,
  INDEX `ENTRY_FK_1` (`hdr_id` ASC) ,
  PRIMARY KEY (`id`) ,
  INDEX `ENTRY_FK_2` (`parent_entry_id` ASC) ,
  CONSTRAINT `ENTRY_FK_1`
    FOREIGN KEY (`hdr_id` )
    REFERENCES `LOGBOOK`.`HEADER` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `ENTRY_FK_2`
    FOREIGN KEY (`parent_entry_id` )
    REFERENCES `LOGBOOK`.`ENTRY` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`TAG`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`TAG` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`TAG` (
  `hdr_id` INT NOT NULL ,
  `tag` VARCHAR(255) NOT NULL ,
  `value` MEDIUMTEXT NULL ,
  INDEX `TAG_FK_1` (`hdr_id` ASC) ,
  PRIMARY KEY (`tag`, `hdr_id`) ,
  CONSTRAINT `TAG_FK_1`
    FOREIGN KEY (`hdr_id` )
    REFERENCES `LOGBOOK`.`HEADER` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`ATTACHMENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`ATTACHMENT` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`ATTACHMENT` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `entry_id` INT NOT NULL ,
  `description` MEDIUMTEXT NOT NULL ,
  `document` LONGBLOB NOT NULL ,
  `document_type` VARCHAR(255) NOT NULL ,
  INDEX `ATTACHMENT_FK_1` (`entry_id` ASC) ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `ATTACHMENT_FK_1`
    FOREIGN KEY (`entry_id` )
    REFERENCES `LOGBOOK`.`ENTRY` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`SHIFT_CREW`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`SHIFT_CREW` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`SHIFT_CREW` (
  `shift_id` INT NOT NULL ,
  `member` VARCHAR(32) NOT NULL ,
  PRIMARY KEY (`shift_id`, `member`) ,
  INDEX `SHIFT_CREW_FK_1` (`shift_id` ASC) ,
  CONSTRAINT `SHIFT_CREW_FK_1`
    FOREIGN KEY (`shift_id` )
    REFERENCES `LOGBOOK`.`SHIFT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`SUBSCRIBER`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`SUBSCRIBER` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`SUBSCRIBER` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `exper_id` INT ,
  `subscriber` VARCHAR(8) NOT NULL ,
  `address` VARCHAR(255) NOT NULL ,
  `subscribed_time` bigint(20) unsigned NOT NULL ,
  `subscribed_host` VARCHAR(255) NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE INDEX `SUBSCRIBER_IDX_1` (`exper_id` ASC, `subscriber` ASC, `address` ASC) )
ENGINE = InnoDB
COMMENT = 'Subscribers who are registered to receive automatic notifications on LogBook updates';




SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
