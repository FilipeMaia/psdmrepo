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
  UNIQUE INDEX `INSTRUMENT_NAME_1` (`name` ASC) )
ENGINE = InnoDB
COMMENT = 'The table to define instruments';


-- -----------------------------------------------------
-- Table `REGDB`.`EXPERIMENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `REGDB`.`EXPERIMENT` ;

CREATE  TABLE IF NOT EXISTS `REGDB`.`EXPERIMENT` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(255) NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  `instr_id` INT NOT NULL ,
  `registration_time` BIGINT NOT NULL ,
  `begin_time` BIGINT UNSIGNED NOT NULL ,
  `end_time` BIGINT UNSIGNED NOT NULL ,
  `leader_account` VARCHAR(32) NOT NULL ,
  `contact_info` VARCHAR(255) NOT NULL ,
  `posix_gid` VARCHAR(45) NOT NULL ,
  PRIMARY KEY (`id`) ,
  INDEX `EXPERIMENT_FK_1` (`instr_id` ASC) ,
  UNIQUE INDEX `EXPERIMENT_NAME_1` (`name` ASC) ,
  CONSTRAINT `EXPERIMENT_FK_1`
    FOREIGN KEY (`instr_id` )
    REFERENCES `REGDB`.`INSTRUMENT` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `REGDB`.`INSTRUMENT_PARAM`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `REGDB`.`INSTRUMENT_PARAM` ;

CREATE  TABLE IF NOT EXISTS `REGDB`.`INSTRUMENT_PARAM` (
  `instr_id` INT NOT NULL ,
  `param` VARCHAR(255) NOT NULL ,
  `val` MEDIUMTEXT NOT NULL ,
  `descr` MEDIUMTEXT NOT NULL ,
  PRIMARY KEY (`instr_id`, `param`) ,
  INDEX `INSTRUMENT_PROP_FK_1` (`instr_id` ASC) ,
  CONSTRAINT `INSTRUMENT_PROP_FK_1`
    FOREIGN KEY (`instr_id` )
    REFERENCES `REGDB`.`INSTRUMENT` (`id` )
    ON DELETE NO ACTION
    ON UPDATE NO ACTION)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `REGDB`.`EXPERIMENT_PARAM`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `REGDB`.`EXPERIMENT_PARAM` ;

CREATE  TABLE IF NOT EXISTS `REGDB`.`EXPERIMENT_PARAM` (
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



SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
