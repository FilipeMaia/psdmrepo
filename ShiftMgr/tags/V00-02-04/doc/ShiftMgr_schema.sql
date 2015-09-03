SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `SHIFTMGR` ;
USE `SHIFTMGR`;


DROP TABLE IF EXISTS `SHIFTMGR`.`SHIFT_AREA_HISTORY` ;
DROP TABLE IF EXISTS `SHIFTMGR`.`SHIFT_AREA_EVALUATION` ;

DROP TABLE IF EXISTS `SHIFTMGR`.`SHIFT_TIME_HISTORY` ;
DROP TABLE IF EXISTS `SHIFTMGR`.`SHIFT_TIME_ALLOCATION` ;

DROP TABLE IF EXISTS `SHIFTMGR`.`SHIFT_HISTORY` ;
DROP TABLE IF EXISTS `SHIFTMGR`.`SHIFT` ;

-- -----------------------------------------------------
-- Table `SHIFTMGR`.`SHIFT`
-- -----------------------------------------------------

CREATE TABLE IF NOT EXISTS `SHIFTMGR`.`SHIFT` (

  `id`         INT             NOT NULL AUTO_INCREMENT ,
  `instr_name` VARCHAR(32)     NOT NULL ,

  `begin_time` BIGINT UNSIGNED NOT NULL ,
  `end_time`   BIGINT UNSIGNED NOT NULL ,

  `notes`      TEXT               DEFAULT '' ,

  `modified_uid`  VARCHAR(32)     DEFAULT NULL ,
  `modified_time` BIGINT UNSIGNED DEFAULT NULL ,

   PRIMARY KEY(`id`) ,
   UNIQUE INDEX `SHIFT_IDX_1` (`instr_name`,`begin_time`)
);

-- -----------------------------------------------------
-- Table `SHIFTMGR`.`SHIFT_AREA_EVALUATION`
-- -----------------------------------------------------

CREATE TABLE IF NOT EXISTS `SHIFTMGR`.`SHIFT_AREA_EVALUATION` (

  `id`       INT NOT NULL AUTO_INCREMENT ,
  `shift_id` INT NOT NULL ,

  `name`         VARCHAR(32)  NOT NULL,
  `problems`     BOOL         NOT NULL,
  `downtime_min` INT UNSIGNED NOT NULL,

  `comments` TEXT DEFAULT '' ,

  PRIMARY KEY(`id`) ,

  CONSTRAINT `SHIFT_AREA_EVALUATION_FK_1`
    FOREIGN KEY (`shift_id` )
    REFERENCES `SHIFTMGR`.`SHIFT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
);

-- -----------------------------------------------------
-- Table `SHIFTMGR`.`SHIFT_TIME_ALLOCATION`
-- -----------------------------------------------------

CREATE TABLE IF NOT EXISTS `SHIFTMGR`.`SHIFT_TIME_ALLOCATION` (

  `id`       INT NOT NULL AUTO_INCREMENT ,
  `shift_id` INT NOT NULL ,

  `name`         VARCHAR(32)  NOT NULL,
  `duration_min` INT UNSIGNED NOT NULL,

  `comments` TEXT DEFAULT '' ,

  PRIMARY KEY(`id`) ,

  CONSTRAINT `SHIFT_TIME_ALLOCATION_FK_1`
    FOREIGN KEY (`shift_id` )
    REFERENCES `SHIFTMGR`.`SHIFT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
);

-- -----------------------------------------------------
-- Table `SHIFTMGR`.`SHIFT_HISTORY`
-- -----------------------------------------------------

CREATE TABLE IF NOT EXISTS `SHIFTMGR`.`SHIFT_HISTORY` (

  `id`       INT NOT NULL AUTO_INCREMENT ,
  `shift_id` INT NOT NULL ,

  `modified_uid`  VARCHAR(32)     NOT NULL ,
  `modified_time` BIGINT UNSIGNED NOT NULL ,

  `event` ENUM('CREATE','MODIFY') NOT NULL ,

  `parameter` TEXT NOT NULL ,
  `old_value` TEXT NOT NULL ,
  `new_value` TEXT NOT NULL ,

  PRIMARY KEY(`id`) ,

  CONSTRAINT `SHIFT_HISTORY_FK_1`
    FOREIGN KEY (`shift_id` )
    REFERENCES `SHIFTMGR`.`SHIFT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
);


-- -----------------------------------------------------
-- Table `SHIFTMGR`.`SHIFT_AREA_HISTORY`
-- -----------------------------------------------------

CREATE TABLE IF NOT EXISTS `SHIFTMGR`.`SHIFT_AREA_HISTORY` (

  `id`      INT NOT NULL AUTO_INCREMENT ,
  `area_id` INT NOT NULL ,

  `modified_uid`  VARCHAR(32)     NOT NULL ,
  `modified_time` BIGINT UNSIGNED NOT NULL ,

  `parameter` TEXT NOT NULL ,
  `old_value` TEXT NOT NULL ,
  `new_value` TEXT NOT NULL ,

  PRIMARY KEY(`id`) ,

  CONSTRAINT `SHIFT_AREA_HISTORY_FK_1`
    FOREIGN KEY (`area_id` )
    REFERENCES `SHIFTMGR`.`SHIFT_AREA_EVALUATION` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
);

-- -----------------------------------------------------
-- Table `SHIFTMGR`.`SHIFT_TIME_HISTORY`
-- -----------------------------------------------------

CREATE TABLE IF NOT EXISTS `SHIFTMGR`.`SHIFT_TIME_HISTORY` (

  `id`      INT NOT NULL AUTO_INCREMENT ,
  `time_id` INT NOT NULL ,

  `modified_uid`  VARCHAR(32)     NOT NULL ,
  `modified_time` BIGINT UNSIGNED NOT NULL ,

  `parameter` TEXT NOT NULL ,
  `old_value` TEXT NOT NULL ,
  `new_value` TEXT NOT NULL ,

  PRIMARY KEY(`id`) ,

  CONSTRAINT `SHIFT_TIME_HISTORY_FK_1`
    FOREIGN KEY (`time_id` )
    REFERENCES `SHIFTMGR`.`SHIFT_TIME_ALLOCATION` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
);


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
