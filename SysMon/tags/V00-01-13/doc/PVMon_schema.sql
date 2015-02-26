SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `SYSMON` ;
USE `SYSMON`;

-- -----------------------------------------------------
-- Table `SYSMON`.`PV`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`PV` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`PV` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(255) NOT NULL ,
  `scope` VARCHAR(255) NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `name_scope` (`name`,`scope`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `SYSMON`.`PV_VAL`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`PV_VAL` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`PV_VAL` (
  `pv_id`     INT             NOT NULL ,
  `timestamp` BIGINT UNSIGNED NOT NULL ,
  `value`     TEXT            NOT NULL ,
  CONSTRAINT `PV_VAL_FK_1`
    FOREIGN KEY (`pv_id` )
    REFERENCES `SYSMON`.`PV` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
