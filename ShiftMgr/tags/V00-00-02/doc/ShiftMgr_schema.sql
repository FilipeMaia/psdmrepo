SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `SHIFTMGR` ;
USE `SHIFTMGR`;

-- -----------------------------------------------------
-- Table `SHIFTMGR`.`SHIFT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SHIFTMGR`.`SHIFT`;
DROP TABLE IF EXISTS `SHIFTMGR`.`AREA_EVALUATION`;
DROP TABLE IF EXISTS `SHIFTMGR`.`TIME_USE`;

CREATE TABLE IF NOT EXISTS `SHIFTMGR`.`SHIFT` (
  `id`                    INT NOT NULL AUTO_INCREMENT ,
  `last_modified_time`    INT NOT NULL,
  `username`              VARCHAR(64) NOT NULL,
  `hutch`                 VARCHAR(32) NOT NULL,
  `start_time`            INT NOT NULL,
  `end_time`              INT DEFAULT NULL,
  `stopper_out`           INT DEFAULT NULL,
  `door_open`             INT DEFAULT NULL,
  `total_shots`           INT DEFAULT NULL,
  `other_notes`           VARCHAR(5000) DEFAULT NULL,
   PRIMARY KEY(`id`)
);

-- -----------------------------------------------------
-- Table `SHIFTMGR`.`AREA_EVALUATION`
-- -----------------------------------------------------

CREATE TABLE IF NOT EXISTS `SHIFTMGR`.`AREA_EVALUATION` (
  `id`                    INT NOT NULL,
  `area`                  VARCHAR(32) NOT NULL,
  `ok`                    BOOL NOT NULL,
  `downtime`              INT NOT NULL,
  `comment`               VARCHAR(5000) DEFAULT NULL,
   PRIMARY KEY(`id`, `area`)
);

-- -----------------------------------------------------
-- Table `SHIFTMGR`.`TIME_USE`
-- -----------------------------------------------------

CREATE TABLE IF NOT EXISTS `SHIFTMGR`.`TIME_USE` (
  `id`                    INT NOT NULL,
  `use_name`              VARCHAR(32) NOT NULL,
  `use_time`              INT NOT NULL,
  `comment`               VARCHAR(5000) DEFAULT NULL,
   PRIMARY KEY(`id`, `use_name`)
);

SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
