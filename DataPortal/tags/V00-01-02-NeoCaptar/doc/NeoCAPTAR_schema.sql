SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `NEOCAPTAR` ;
USE `NEOCAPTAR`;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`DICT_CABLE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`DICT_CABLE` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`DICT_CABLE` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(255) NOT NULL ,
  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid` VARCHAR(32) NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `name` (`name`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`DICT_CONNECTOR`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`DICT_CONNECTOR` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`DICT_CONNECTOR` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `cable_id` INT NOT NULL ,
  `name` VARCHAR(255) NOT NULL ,
  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid` VARCHAR(32) NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `name` (`cable_id`,`name`) ,
  INDEX `DICT_CONNECTOR_FK_1` (`cable_id` ASC) ,
  CONSTRAINT `DICT_CONNECTOR_FK_1`
    FOREIGN KEY (`cable_id` )
    REFERENCES `NEOCAPTAR`.`DICT_CABLE` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`DICT_PINLIST`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`DICT_PINLIST` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`DICT_PINLIST` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `connector_id` INT NOT NULL ,
  `name` VARCHAR(255) NOT NULL ,
  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid` VARCHAR(32) NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `name` (`connector_id`,`name`) ,
  INDEX `DICT_PINLIST_FK_1` (`connector_id` ASC) ,
  CONSTRAINT `DICT_PINLIST_FK_1`
    FOREIGN KEY (`connector_id` )
    REFERENCES `NEOCAPTAR`.`DICT_CONNECTOR` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`DICT_LOCATION`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`DICT_LOCATION` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`DICT_LOCATION` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(255) NOT NULL ,
  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid` VARCHAR(32) NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `name` (`name`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`DICT_RACK`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`DICT_RACK` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`DICT_RACK` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `location_id` INT NOT NULL ,
  `name` VARCHAR(255) NOT NULL ,
  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid` VARCHAR(32) NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `name` (`location_id`,`name`) ,
  INDEX `DICT_RACK_FK_1` (`location_id` ASC) ,
  CONSTRAINT `DICT_RACK_FK_1`
    FOREIGN KEY (`location_id` )
    REFERENCES `NEOCAPTAR`.`DICT_LOCATION` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;



SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
