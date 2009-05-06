SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

DROP SCHEMA `SCIMD` ;
CREATE SCHEMA IF NOT EXISTS `SCIMD` ;
USE `SCIMD`;

-- -----------------------------------------------------
-- Table `SCIMD`.`INSTRUMENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`INSTRUMENT` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`INSTRUMENT` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(255) NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE INDEX `INSTRUMENT_IDX_1` (`name` ASC) )
ENGINE = InnoDB
COMMENT = 'The table to define instruments';


-- -----------------------------------------------------
-- Table `SCIMD`.`GROUP`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`GROUP` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`GROUP` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(255) NOT NULL ,
  `proj_id` VARCHAR(255) NULL ,
  `contact` VARCHAR(255) NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE INDEX `GROUP_IDX_1` (`name` ASC) )
ENGINE = InnoDB
COMMENT = 'A scientific group participating in experiments';


-- -----------------------------------------------------
-- Table `SCIMD`.`EXPERIMENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`EXPERIMENT` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`EXPERIMENT` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(255) NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  `instr_id` INT NOT NULL ,
  `group_id` INT NOT NULL ,
  `begin_time` DATETIME NOT NULL ,
  `end_time` DATETIME NULL ,
  PRIMARY KEY (`id`) ,
  INDEX `EXPERIMENT_FK_1` (`instr_id` ASC) ,
  INDEX `EXPERIMENT_FK_2` (`group_id` ASC) ,
  UNIQUE INDEX `EXPERIMENT_IDX_1` (`name` ASC) ,
  CONSTRAINT `EXPERIMENT_FK_1`
    FOREIGN KEY (`instr_id` )
    REFERENCES `SCIMD`.`INSTRUMENT` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `EXPERIMENT_FK_2`
    FOREIGN KEY (`group_id` )
    REFERENCES `SCIMD`.`GROUP` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `SCIMD`.`INSTRUMENT_PROP`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`INSTRUMENT_PROP` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`INSTRUMENT_PROP` (
  `instr_id` INT NOT NULL ,
  `param` VARCHAR(255) NOT NULL ,
  `val` MEDIUMTEXT NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  PRIMARY KEY (`instr_id`, `param`) ,
  INDEX `INSTRUMENT_PROP_FK_1` (`instr_id` ASC) ,
  UNIQUE INDEX `INSTRUMENT_IDX_1` (`param` ASC) ,
  CONSTRAINT `INSTRUMENT_PROP_FK_1`
    FOREIGN KEY (`instr_id` )
    REFERENCES `SCIMD`.`INSTRUMENT` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `SCIMD`.`EXPERIMENT_CONF`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`EXPERIMENT_CONF` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`EXPERIMENT_CONF` (
  `exper_id` INT NOT NULL ,
  `param` VARCHAR(255) NOT NULL ,
  `val` MEDIUMTEXT NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  PRIMARY KEY (`exper_id`, `param`) ,
  INDEX `EXPERIMENT_CONF_FK_1` (`exper_id` ASC) ,
  CONSTRAINT `EXPERIMENT_CONF_FK_1`
    FOREIGN KEY (`exper_id` )
    REFERENCES `SCIMD`.`EXPERIMENT` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Optional configuration parameters of an experiment';


-- -----------------------------------------------------
-- Table `SCIMD`.`RUN`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`RUN` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`RUN` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `num` INT NOT NULL ,
  `exper_id` INT NOT NULL ,
  `type` ENUM('DATA','CALIB') NOT NULL DEFAULT 'DATA' ,
  `begin_time` DATETIME NOT NULL ,
  `end_time` DATETIME NOT NULL ,
  INDEX `RUN_FK_1` (`exper_id` ASC) ,
  UNIQUE INDEX `RUN_IDX_1` (`exper_id` ASC, `num` ASC) ,
  PRIMARY KEY (`id`) ,
  CONSTRAINT `RUN_FK_1`
    FOREIGN KEY (`exper_id` )
    REFERENCES `SCIMD`.`EXPERIMENT` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `SCIMD`.`RUN_PARAM`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`RUN_PARAM` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`RUN_PARAM` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `param` VARCHAR(255) NOT NULL ,
  `exper_id` INT NOT NULL ,
  `type` ENUM('INT','DOUBLE','TEXT') NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  PRIMARY KEY (`id`) ,
  INDEX `RUN_PARAM_FK_1` (`exper_id` ASC) ,
  UNIQUE INDEX `RUN_PARAM_IDX_1` (`param` ASC, `exper_id` ASC) ,
  CONSTRAINT `RUN_PARAM_FK_1`
    FOREIGN KEY (`exper_id` )
    REFERENCES `SCIMD`.`EXPERIMENT` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
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
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `RUN_VAL_INT_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `SCIMD`.`RUN_PARAM` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
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
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `RUN_VAL_TEXT_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `SCIMD`.`RUN_PARAM` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
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
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `RUN_VAL_DOUBLE_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `SCIMD`.`RUN_PARAM` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `SCIMD`.`EXPER_X_RUN_PARAM`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`EXPER_X_RUN_PARAM` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`EXPER_X_RUN_PARAM` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `param` VARCHAR(255) NOT NULL ,
  `type` ENUM('INT','DOUBLE','TEXT') NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  PRIMARY KEY (`id`) )
ENGINE = InnoDB
COMMENT = 'Extended key names and types for variable parameters of runs';


-- -----------------------------------------------------
-- Table `SCIMD`.`EXPER_X_RUN_VAL_DOUBLE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`EXPER_X_RUN_VAL_DOUBLE` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`EXPER_X_RUN_VAL_DOUBLE` (
  `run_id` INT NOT NULL AUTO_INCREMENT ,
  `param_id` INT NOT NULL ,
  `val` DOUBLE NULL ,
  PRIMARY KEY (`run_id`, `param_id`) ,
  INDEX `EXPER_X_RUN_VAL_DOUBLE_FK_1` (`run_id` ASC) ,
  INDEX `EXPER_X_RUN_VAL_DOUBLE_FK_2` (`param_id` ASC) ,
  CONSTRAINT `EXPER_X_RUN_VAL_DOUBLE_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `SCIMD`.`RUN` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `EXPER_X_RUN_VAL_DOUBLE_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `SCIMD`.`EXPER_X_RUN_PARAM` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `SCIMD`.`EXPER_X_RUN_VAL_INT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`EXPER_X_RUN_VAL_INT` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`EXPER_X_RUN_VAL_INT` (
  `run_id` INT NOT NULL AUTO_INCREMENT ,
  `param_id` INT NOT NULL ,
  `val` INT NULL ,
  PRIMARY KEY (`run_id`, `param_id`) ,
  INDEX `EXPER_X_RUN_VAL_INT_FK_1` (`run_id` ASC) ,
  INDEX `EXPER_X_RUN_VAL_INT_FK_2` (`param_id` ASC) ,
  CONSTRAINT `EXPER_X_RUN_VAL_INT_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `SCIMD`.`RUN` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `EXPER_X_RUN_VAL_INT_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `SCIMD`.`EXPER_X_RUN_PARAM` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `SCIMD`.`EXPER_X_RUN_VAL_TEXT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`EXPER_X_RUN_VAL_TEXT` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`EXPER_X_RUN_VAL_TEXT` (
  `run_id` INT NOT NULL AUTO_INCREMENT ,
  `param_id` INT NOT NULL ,
  `val` MEDIUMTEXT NULL ,
  PRIMARY KEY (`run_id`, `param_id`) ,
  INDEX `EXPER_X_RUN_VAL_TEXT_FK_1` (`run_id` ASC) ,
  INDEX `EXPER_X_RUN_VAL_TEXT_FK_2` (`param_id` ASC) ,
  CONSTRAINT `EXPER_X_RUN_VAL_TEXT_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `SCIMD`.`RUN` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `EXPER_X_RUN_VAL_TEXT_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `SCIMD`.`EXPER_X_RUN_PARAM` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `SCIMD`.`RUN_VAL`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SCIMD`.`RUN_VAL` ;

CREATE  TABLE IF NOT EXISTS `SCIMD`.`RUN_VAL` (
  `run_id` INT NOT NULL ,
  `param_id` INT NOT NULL ,
  `source` VARCHAR(255) NOT NULL ,
  `updated` DATETIME NOT NULL ,
  INDEX `RUN_VAL_FK_1` (`run_id` ASC) ,
  INDEX `RUN_VAL_FK_2` (`param_id` ASC) ,
  PRIMARY KEY (`param_id`, `run_id`) ,
  CONSTRAINT `RUN_VAL_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `SCIMD`.`RUN` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `RUN_VAL_FK_2`
    FOREIGN KEY (`param_id` )
    REFERENCES `SCIMD`.`RUN_PARAM` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;



SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
