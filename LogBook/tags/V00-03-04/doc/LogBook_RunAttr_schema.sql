SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `LOGBOOK` ;
USE `LOGBOOK`;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`RUN_ATTR`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`RUN_ATTR` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`RUN_ATTR` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `run_id` INT NOT NULL ,
  `class` VARCHAR(255) NOT NULL ,
  `name` VARCHAR(255) NOT NULL ,
  `type` ENUM('INT','DOUBLE','TEXT') NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE INDEX `RUN_ATTR_IDX_1` (`run_id` ASC, `class` ASC, `name` ASC) ,
  CONSTRAINT `RUN_ATTR_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `LOGBOOK`.`RUN` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE )
ENGINE = InnoDB
COMMENT = 'Key names and types for attributes of runs';


-- -----------------------------------------------------
-- Table `LOGBOOK`.`RUN_ATTR_INT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`RUN_ATTR_INT` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`RUN_ATTR_INT` (
  `attr_id` INT NOT NULL ,
  `val` INT NULL ,
  PRIMARY KEY (`attr_id`) ,
  CONSTRAINT `RUN_ATTR_INT_FK_1`
    FOREIGN KEY (`attr_id` )
    REFERENCES `LOGBOOK`.`RUN_ATTR` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`RUN_ATTR_DOUBLE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`RUN_ATTR_DOUBLE` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`RUN_ATTR_DOUBLE` (
  `attr_id` INT NOT NULL ,
  `val` DOUBLE NULL ,
  PRIMARY KEY (`attr_id`) ,
  CONSTRAINT `RUN_ATTR_DOUBLE_FK_1`
    FOREIGN KEY (`attr_id` )
    REFERENCES `LOGBOOK`.`RUN_ATTR` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`RUN_ATTR_TEXT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`RUN_ATTR_TEXT` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`RUN_ATTR_TEXT` (
  `attr_id` INT NOT NULL ,
  `val` MEDIUMTEXT NULL ,
  PRIMARY KEY (`attr_id`) ,
  CONSTRAINT `RUN_ATTR_TEXT_FK_1`
    FOREIGN KEY (`attr_id` )
    REFERENCES `LOGBOOK`.`RUN_ATTR` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
