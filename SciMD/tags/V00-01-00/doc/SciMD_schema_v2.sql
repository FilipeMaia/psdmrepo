SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `SCIMD` ;
USE `SCIMD`;

-- -----------------------------------------------------
-- Table `SCIMD`.`RUN`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`RUN` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`RUN` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `num` INT NOT NULL ,
  `exper_id` INT NOT NULL ,
  `type` ENUM('DATA','CALIB') NOT NULL DEFAULT 'DATA' ,
  `begin_time` BIGINT UNSIGNED NOT NULL ,
  `end_time` BIGINT UNSIGNED NOT NULL ,
  UNIQUE INDEX `RUN_IDX_1` (`exper_id` ASC, `num` ASC) ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `SCIMD`.`RUN_PARAM`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`RUN_PARAM` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`RUN_PARAM` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `param` VARCHAR(255) NOT NULL ,
  `exper_id` INT NOT NULL ,
  `type` ENUM('INT','INT64','DOUBLE','TEXT') NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE INDEX `RUN_PARAM_IDX_1` (`param` ASC, `exper_id` ASC) )
ENGINE = InnoDB
COMMENT = 'Key names and types for variable parameters of runs';


-- -----------------------------------------------------
-- Table `SCIMD`.`RUN_VAL_INT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`RUN_VAL_INT` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`RUN_VAL_INT` (
  `run_id` INT NOT NULL AUTO_INCREMENT ,
  `param_id` INT NOT NULL ,
  `val` INT NULL ,
  PRIMARY KEY (`run_id`, `param_id`) ,
  INDEX `RUN_VAL_INT_FK_1` (`run_id` ASC) ,
  INDEX `RUN_VAL_INT_FK_2` (`param_id` ASC) ,
  CONSTRAINT `RUN_VAL_INT_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `SCIMD`.`RUN` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `RUN_VAL_INT_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `SCIMD`.`RUN_PARAM` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `SCIMD`.`RUN_VAL_TEXT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`RUN_VAL_TEXT` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`RUN_VAL_TEXT` (
  `run_id` INT NOT NULL AUTO_INCREMENT ,
  `param_id` INT NOT NULL ,
  `val` MEDIUMTEXT NULL ,
  INDEX `RUN_VAL_TEXT_FK_1` (`run_id` ASC) ,
  INDEX `RUN_VAL_TEXT_FK_2` (`param_id` ASC) ,
  PRIMARY KEY (`run_id`, `param_id`) ,
  CONSTRAINT `RUN_VAL_TEXT_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `SCIMD`.`RUN` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `RUN_VAL_TEXT_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `SCIMD`.`RUN_PARAM` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `SCIMD`.`RUN_VAL_DOUBLE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`RUN_VAL_DOUBLE` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`RUN_VAL_DOUBLE` (
  `run_id` INT NOT NULL AUTO_INCREMENT ,
  `param_id` INT NOT NULL ,
  `val` DOUBLE NULL ,
  INDEX `RUN_VAL_DOUBLE_FK_1` (`run_id` ASC) ,
  INDEX `RUN_VAL_DOUBLE_FK_2` (`param_id` ASC) ,
  PRIMARY KEY (`run_id`, `param_id`) ,
  CONSTRAINT `RUN_VAL_DOUBLE_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `SCIMD`.`RUN` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `RUN_VAL_DOUBLE_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `SCIMD`.`RUN_PARAM` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `SCIMD`.`RUN_VAL`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`RUN_VAL` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`RUN_VAL` (
  `run_id` INT NOT NULL ,
  `param_id` INT NOT NULL ,
  `source` VARCHAR(255) NOT NULL ,
  `updated` BIGINT UNSIGNED NOT NULL ,
  INDEX `RUN_VAL_FK_1` (`run_id` ASC) ,
  INDEX `RUN_VAL_FK_2` (`param_id` ASC) ,
  PRIMARY KEY (`param_id`, `run_id`) ,
  CONSTRAINT `RUN_VAL_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `SCIMD`.`RUN` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `RUN_VAL_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `SCIMD`.`RUN_PARAM` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `SCIMD`.`RUN_VAL_INT64`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`RUN_VAL_INT64` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`RUN_VAL_INT64` (
  `run_id` INT NOT NULL AUTO_INCREMENT ,
  `param_id` INT NOT NULL ,
  `val` INT NULL ,
  PRIMARY KEY (`run_id`, `param_id`) ,
  INDEX `RUN_VAL_INT64_FK_1` (`run_id` ASC) ,
  INDEX `RUN_VAL_INT64_FK_2` (`param_id` ASC) ,
  CONSTRAINT `RUN_VAL_INT64_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `SCIMD`.`RUN` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE,
  CONSTRAINT `RUN_VAL_INT64_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `SCIMD`.`RUN_PARAM` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;



SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
