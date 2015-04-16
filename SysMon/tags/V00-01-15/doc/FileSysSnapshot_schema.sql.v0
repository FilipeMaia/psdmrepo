SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `SYSMON` ;
USE `SYSMON`;

-- -----------------------------------------------------
-- Table `SYSMON`.`FILE_SYSTEM`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`FILE_SYSTEM` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`FILE_SYSTEM` (

  `id` INT NOT NULL AUTO_INCREMENT ,

  `base_path` VARCHAR(512) NOT NULL ,
  `scan_time` INT UNSIGNED NOT NULL ,

  PRIMARY KEY (`id`))

ENGINE = InnoDB ;

-- -----------------------------------------------------
-- Table `SYSMON`.`FILE_TYPE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`FILE_TYPE` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`FILE_TYPE` (

  `id`             INT NOT NULL AUTO_INCREMENT ,
  `file_system_id` INT NOT NULL ,

  `name` VARCHAR(512) NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE KEY `name` (`file_system_id`,`name`) ,

  CONSTRAINT `FILE_TYPE_FK_1`
    FOREIGN KEY (`file_system_id` )
    REFERENCES `SYSMON`.`FILE_SYSTEM` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)

ENGINE = InnoDB ;


-- -----------------------------------------------------
-- Table `SYSMON`.`FILE_CATALOG`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`FILE_CATALOG` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`FILE_CATALOG` (

  `id`             INT NOT NULL AUTO_INCREMENT ,
  `parent_id`      INT     NULL ,
  `file_system_id` INT NOT NULL ,

  `name`         VARCHAR(512)                              NOT NULL ,
  `entry_type`   ENUM('DIR','LINK','FILE','OTHER','ERROR') NOT NULL ,
  `extension`    VARCHAR(512)                                  NULL ,  -- regular files only
  `file_type_id` INT                                           NULL ,  -- regular files only

  `owner_uid`     INT UNSIGNED NULL ,  -- all but 'ERROR'
  `owner_gid`     INT UNSIGNED NULL ,  -- all but 'ERROR'
  `size_bytes` BIGINT UNSIGNED NULL ,  -- regular files only
  `atime`         INT UNSIGNED NULL ,  -- regular files only
  `ctime`         INT UNSIGNED NULL ,  -- regular files only
  `mtime`         INT UNSIGNED NULL ,  -- regular files only

  PRIMARY KEY (`id`) ,

  UNIQUE KEY `name` (parent_id,file_system_id,`name`) ,

  CONSTRAINT `FILE_CATALOG_FK_1`
    FOREIGN KEY (`parent_id` )
    REFERENCES `SYSMON`.`FILE_CATALOG` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE ,

  CONSTRAINT `FILE_CATALOG_FK_2`
    FOREIGN KEY (`file_system_id` )
    REFERENCES `SYSMON`.`FILE_SYSTEM` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE ,

  CONSTRAINT `FILE_CATALOG_FK_3`
    FOREIGN KEY (`file_type_id` )
    REFERENCES `SYSMON`.`FILE_TYPE` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE )

ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `SYSMON`.`FILE_ACTION`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`FILE_ACTION` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`FILE_ACTION` (

  `file_id`      INT     NULL ,

  `action_time`  INT UNSIGNED NULL ,
  `result_type`  ENUM('DELETED','ERROR','FILE_IS_CURRENT','NO_FILE_FOUND') NOT NULL ,

  CONSTRAINT `FILE_ACTION_FK_1`
    FOREIGN KEY (`file_id` )
    REFERENCES `SYSMON`.`FILE_CATALOG` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE )

ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
