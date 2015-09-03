SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `SYSMON` ;
USE `SYSMON`;

-- -----------------------------------------------------
-- Table `SYSMON`.`BEAMTIME_CONFIG`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`BEAMTIME_CONFIG` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`BEAMTIME_CONFIG` (
  `param` VARCHAR(255) NOT NULL ,
  `value` TEXT ,
  UNIQUE KEY `param` (`param`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `SYSMON`.`BEAMTIME_RUNS`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`BEAMTIME_RUNS` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`BEAMTIME_RUNS` (
  `begin_time` BIGINT UNSIGNED NOT NULL ,
  `end_time`   BIGINT UNSIGNED NOT NULL ,
  `exper_id`   INT             NOT NULL ,
  `runnum`     INT             NOT NULL ,
  `exper_name` VARCHAR(255)    NOT NULL ,
  `instr_name` VARCHAR(255)    NOT NULL ,
  INDEX `BEAMTIME_RUNS_BEGIN_TIME` (`begin_time`) ,
  INDEX `BEAMTIME_RUNS_END_TIME`   (`begin_time`) ,
  UNIQUE KEY `run`                 (`exper_id`, `runnum`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `SYSMON`.`BEAMTIME_GAPS`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`BEAMTIME_GAPS` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`BEAMTIME_GAPS` (
  `begin_time` BIGINT UNSIGNED NOT NULL ,
  `end_time`   BIGINT UNSIGNED NOT NULL ,
  `instr_name` VARCHAR(255)    NOT NULL ,
  INDEX `BEAMTIME_GAPS_BEGIN_TIME` (`begin_time`) ,
  INDEX `BEAMTIME_GAPS_END_TIME`   (`begin_time`) ,
  INDEX `BEAMTIME_GAPS_INSTR`      (`instr_name`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `SYSMON`.`BEAMTIME_COMMENTS`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`BEAMTIME_COMMENTS` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`BEAMTIME_COMMENTS` (
  `gap_begin_time` BIGINT UNSIGNED NOT NULL ,
  `instr_name`     VARCHAR(255)    NOT NULL ,
  `post_time`      BIGINT UNSIGNED NOT NULL ,
  `posted_by_uid`  VARCHAR(32)     NOT NULL ,
  `comment`        TEXT            NOT NULL ,
  `system`         VARCHAR(255)    NOT NULL ,
  UNIQUE KEY `gap_begin_time_instr_name`   (`gap_begin_time`,`instr_name`) ,
  INDEX `BEAMTIME_COMMENTS_GAP_BEGIN_TIME` (`gap_begin_time`))
ENGINE = InnoDB ;

-- -----------------------------------------------------
-- Table `SYSMON`.`BEAMTIME_SUBSCRIBER`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`BEAMTIME_SUBSCRIBER` ;

CREATE TABLE IF NOT EXISTS `SYSMON`.`BEAMTIME_SUBSCRIBER` (
  `id`              INT             NOT NULL AUTO_INCREMENT ,
  `subscriber`      VARCHAR(32)     NOT NULL ,
  `address`         VARCHAR(255)    NOT NULL ,
  `subscribed_by`   VARCHAR(32)     NOT NULL ,
  `subscribed_time` BIGINT UNSIGNED NOT NULL ,
  `subscribed_host` varchar(255)    NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `BEAMTIME_SUBSCRIBER_IDX_1` (`subscriber`,`address`)
) ENGINE = InnoDB ;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
