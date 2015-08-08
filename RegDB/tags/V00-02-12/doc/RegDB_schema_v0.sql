SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `REGDB` ;
USE `REGDB`;

-- -----------------------------------------------------
-- Table `REGDB`.`INSTRUMENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `REGDB`.`INSTRUMENT` ;

CREATE  TABLE IF NOT EXISTS `REGDB`.`INSTRUMENT` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(255) NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE INDEX `INSTRUMENT_IDX_1` (`name` ASC) )
ENGINE = InnoDB
COMMENT = 'The table to define instruments';


-- -----------------------------------------------------
-- Table `REGDB`.`GROUP`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `REGDB`.`GROUP` ;

CREATE  TABLE IF NOT EXISTS `REGDB`.`GROUP` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(255) NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  `proj_id` VARCHAR(255) NULL ,
  `contact` VARCHAR(255) NOT NULL ,
  `leader` VARCHAR(32) NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE INDEX `GROUP_IDX_1` (`name` ASC) )
ENGINE = InnoDB
COMMENT = 'A scientific group participating in experiments';


-- -----------------------------------------------------
-- Table `REGDB`.`EXPERIMENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `REGDB`.`EXPERIMENT` ;

CREATE  TABLE IF NOT EXISTS `REGDB`.`EXPERIMENT` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(255) NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  `instr_id` INT NOT NULL ,
  `group_id` INT NOT NULL ,
  `registered` BIGINT NOT NULL ,
  `begin` BIGINT UNSIGNED NOT NULL ,
  `end` BIGINT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) ,
  INDEX `EXPERIMENT_FK_1` (`instr_id` ASC) ,
  INDEX `EXPERIMENT_FK_2` (`group_id` ASC) ,
  UNIQUE INDEX `EXPERIMENT_IDX_1` (`name` ASC) ,
  CONSTRAINT `EXPERIMENT_FK_1`
    FOREIGN KEY (`instr_id` )
    REFERENCES `REGDB`.`INSTRUMENT` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION,
  CONSTRAINT `EXPERIMENT_FK_2`
    FOREIGN KEY (`group_id` )
    REFERENCES `REGDB`.`GROUP` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `REGDB`.`INSTRUMENT_PROPERTY`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `REGDB`.`INSTRUMENT_PROPERTY` ;

CREATE  TABLE IF NOT EXISTS `REGDB`.`INSTRUMENT_PROPERTY` (
  `instr_id` INT NOT NULL ,
  `param` VARCHAR(255) NOT NULL ,
  `val` MEDIUMTEXT NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  PRIMARY KEY (`instr_id`, `param`) ,
  INDEX `INSTRUMENT_PROP_FK_1` (`instr_id` ASC) ,
  UNIQUE INDEX `INSTRUMENT_IDX_1` (`param` ASC) ,
  CONSTRAINT `INSTRUMENT_PROP_FK_1`
    FOREIGN KEY (`instr_id` )
    REFERENCES `REGDB`.`INSTRUMENT` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `REGDB`.`EXPERIMENT_CONFIG`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `REGDB`.`EXPERIMENT_CONFIG` ;

CREATE  TABLE IF NOT EXISTS `REGDB`.`EXPERIMENT_CONFIG` (
  `exper_id` INT NOT NULL ,
  `param` VARCHAR(255) NOT NULL ,
  `val` MEDIUMTEXT NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  PRIMARY KEY (`exper_id`, `param`) ,
  INDEX `EXPERIMENT_CONF_FK_1` (`exper_id` ASC) ,
  CONSTRAINT `EXPERIMENT_CONF_FK_1`
    FOREIGN KEY (`exper_id` )
    REFERENCES `REGDB`.`EXPERIMENT` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB
COMMENT = 'Optional configuration parameters of an experiment';


-- -----------------------------------------------------
-- Table `REGDB`.`GROUP_MEMBER`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `REGDB`.`GROUP_MEMBER` ;

CREATE  TABLE IF NOT EXISTS `REGDB`.`GROUP_MEMBER` (
  `group_id` INT NOT NULL ,
  `member` VARCHAR(32) NOT NULL ,
  `first_name` VARCHAR(255) NOT NULL ,
  `last_name` VARCHAR(255) NOT NULL ,
  `middle_name` VARCHAR(255) NULL ,
  `phone` VARCHAR(255) NULL ,
  `email` VARCHAR(255) NULL ,
  PRIMARY KEY (`group_id`, `member`) ,
  INDEX `GROUP_MEMBER_FK_1` (`group_id` ASC) ,
  CONSTRAINT `GROUP_MEMBER_FK_1`
    FOREIGN KEY (`group_id` )
    REFERENCES `REGDB`.`GROUP` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;



SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
