SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `REGDB` ;
USE `REGDB`;

-- -----------------------------------------------------
-- Table `REGDB`.`DATA_MIGRATION`
-- -----------------------------------------------------

DROP TABLE IF EXISTS `REGDB`.`DATA_MIGRATION` ;

CREATE  TABLE IF NOT EXISTS `REGDB`.`DATA_MIGRATION` (
  `exper_id` INT NOT NULL ,
  `file` VARCHAR(255) NOT NULL ,
  `file_type` TEXT NOT NULL ,
  `start_time` BIGINT UNSIGNED DEFAULT NULL ,
  `stop_time` BIGINT UNSIGNED DEFAULT NULL ,
  `error_msg` TEXT ,
  `host` TEXT NOT NULL ,
  `dirpath` TEXT NOT NULL ,
  PRIMARY KEY (`exper_id`,`file`) ,
  INDEX `EXPSWITCH_FK_1` (`exper_id` ASC) ,
  CONSTRAINT `DATA_MIGRATION_FK_1`
    FOREIGN KEY (`exper_id` )
    REFERENCES `REGDB`.`EXPERIMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Trigger `REGDB`.`REGDB_FILE_INSERT`
-- -----------------------------------------------------

DROP TRIGGER IF EXISTS `REGDB`.`REGDB_FILE_INSERT`;

DELIMITER |
CREATE TRIGGER `REGDB`.`REGDB_FILE_INSERT` AFTER INSERT ON `REGDB`.`FILE`
  FOR EACH ROW
  BEGIN

    INSERT INTO `REGDB`.`DATA_MIGRATION`
    VALUES(
      NEW.exper_id ,
      CONCAT('e',NEW.exper_id,'-r',LPAD(NEW.run,4,'0000'),'-s',LPAD(NEW.stream,2,'00'),'-c',LPAD(NEW.chunk,2,'00'),'.xtc') ,
      'xtc' ,
      NULL ,
      NULL ,
      NULL ,
      NEW.host ,
      REVERSE(SUBSTR(REVERSE(NEW.dirpath),LOCATE('/',REVERSE(NEW.dirpath))+1))) ;

    INSERT INTO `REGDB`.`DATA_MIGRATION`
    VALUES(
      NEW.exper_id ,
      CONCAT('e',NEW.exper_id,'-r',LPAD(NEW.run,4,'0000'),'-s',LPAD(NEW.stream,2,'00'),'-c',LPAD(NEW.chunk,2,'00'),'.xtc.idx') ,
      'xtc.idx' ,
      NULL ,
      NULL ,
      NULL ,
      NEW.host ,
      CONCAT(REVERSE(SUBSTR(REVERSE(NEW.dirpath),LOCATE('/',REVERSE(NEW.dirpath))+1)),'/index')) ;

  END; |
DELIMITER ;



-- -----------------------------------------------------
-- Table `REGDB`.`DATA_MIGRATION_FFB`
-- -----------------------------------------------------

DROP TABLE IF EXISTS `REGDB`.`DATA_MIGRATION_FFB` ;

CREATE  TABLE IF NOT EXISTS `REGDB`.`DATA_MIGRATION_FFB` (
  `exper_id` INT NOT NULL ,
  `file` VARCHAR(255) NOT NULL ,
  `file_type` TEXT NOT NULL ,
  `start_time` BIGINT UNSIGNED DEFAULT NULL ,
  `stop_time` BIGINT UNSIGNED DEFAULT NULL ,
  `error_msg` TEXT ,
  `host` TEXT NOT NULL ,
  `dirpath` TEXT NOT NULL ,
  PRIMARY KEY (`exper_id`,`file`) ,
  INDEX `DATA_MIGRATION_FFB_FK_1` (`exper_id` ASC) ,
  CONSTRAINT `DATA_MIGRATION_FFB_FK_1`
    FOREIGN KEY (`exper_id` )
    REFERENCES `REGDB`.`EXPERIMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Trigger `REGDB`.`REGDB_FILE_INSERT_FFB`
-- -----------------------------------------------------

DROP TRIGGER IF EXISTS `REGDB`.`REGDB_FILE_INSERT_FFB`;

DELIMITER |
CREATE TRIGGER `REGDB`.`REGDB_FILE_INSERT_FFB` AFTER INSERT ON `REGDB`.`FILE_FFB`
  FOR EACH ROW
  BEGIN

    INSERT INTO `REGDB`.`DATA_MIGRATION_FFB`
    VALUES(
      NEW.exper_id ,
      CONCAT('e',NEW.exper_id,'-r',LPAD(NEW.run,4,'0000'),'-s',LPAD(NEW.stream,2,'00'),'-c',LPAD(NEW.chunk,2,'00'),'.xtc') ,
      'xtc' ,
      NULL ,
      NULL ,
      NULL ,
      NEW.host ,
      REVERSE(SUBSTR(REVERSE(NEW.dirpath),LOCATE('/',REVERSE(NEW.dirpath))+1))) ;

    INSERT INTO `REGDB`.`DATA_MIGRATION_FFB`
    VALUES(
      NEW.exper_id ,
      CONCAT('e',NEW.exper_id,'-r',LPAD(NEW.run,4,'0000'),'-s',LPAD(NEW.stream,2,'00'),'-c',LPAD(NEW.chunk,2,'00'),'.xtc.idx') ,
      'xtc.idx' ,
      NULL ,
      NULL ,
      NULL ,
      NEW.host ,
      CONCAT(REVERSE(SUBSTR(REVERSE(NEW.dirpath),LOCATE('/',REVERSE(NEW.dirpath))+1)),'/index')) ;

  END; |
DELIMITER ;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
