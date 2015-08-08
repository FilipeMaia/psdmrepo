SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `SYSMON` ;
USE `SYSMON`;

-- -----------------------------------------------------
-- Table `SYSMON`.`fs_mon_def`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`FS_MON_DEF` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`FS_MON_DEF` (

  `id` INT NOT NULL AUTO_INCREMENT ,

  `group` VARCHAR(255)  NOT NULL ,  -- a logical group to which the file system belongs to
  `name`  VARCHAR(255)  NOT NULL ,  -- the name of the file system

  PRIMARY KEY (`id`) ,
  UNIQUE KEY `group_name` (`group`,`name`)
)
ENGINE  = InnoDB
COMMENT = 'File system definitions for the monitoring purposes';

-- -----------------------------------------------------
-- Table `SYSMON`.`fs_mon_stat`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`FS_MON_STAT` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`FS_MON_STAT` (

  `fs_id` INT NOT NULL ,

  `insert_time` INT UNSIGNED  NOT NULL ,   -- when the information was reported into the database

  `used`       BIGINT UNSIGNED NOT NULL ,  -- the total number of bytes used
  `available`  BIGINT UNSIGNED NOT NULL ,  -- the total number of bytes which are still available

  CONSTRAINT `FS_MON_STAT_FK_1`
    FOREIGN KEY (`fs_id` )
    REFERENCES `SYSMON`.`FS_MON_DEF` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE  = InnoDB
COMMENT = 'Actaul measurements for the file system usage';


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

