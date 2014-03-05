SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `REGDB` ;
USE `REGDB`;

-- -----------------------------------------------------
-- Table `REGDB`.`FILE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `REGDB`.`FILE` ;

CREATE  TABLE IF NOT EXISTS `REGDB`.`FILE` (
  `exper_id` INT NOT NULL ,
  `run` INT NOT NULL,
  `stream` INT NOT NULL,
  `chunk` INT NOT NULL,
  `open` BIGINT UNSIGNED NOT NULL ,
  `host` VARCHAR(255) NOT NULL ,
  `dirpath` VARCHAR(255) NOT NULL ,
  PRIMARY KEY (`exper_id`,`run`,`stream`,`chunk`) ,
  KEY `FILE_FK_1` (`exper_id`) ,
  CONSTRAINT `FILE_FK_1`
    FOREIGN KEY (`exper_id` )
    REFERENCES `REGDB`.`EXPERIMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `REGDB`.`FILE_FFB`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `REGDB`.`FILE_FFB` ;

CREATE  TABLE IF NOT EXISTS `REGDB`.`FILE_FFB` (
  `exper_id` INT NOT NULL ,
  `run` INT NOT NULL,
  `stream` INT NOT NULL,
  `chunk` INT NOT NULL,
  `open` BIGINT UNSIGNED NOT NULL ,
  `host` VARCHAR(255) NOT NULL ,
  `dirpath` VARCHAR(255) NOT NULL ,
  PRIMARY KEY (`exper_id`,`run`,`stream`,`chunk`) ,
  KEY `FILE_FFB_FK_1` (`exper_id`) ,
  CONSTRAINT `FILE_FFB_FK_1`
    FOREIGN KEY (`exper_id` )
    REFERENCES `REGDB`.`EXPERIMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
