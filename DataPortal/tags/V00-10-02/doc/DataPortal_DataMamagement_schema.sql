SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `WEBPORTAL` ;
USE `WEBPORTAL`;

-- -----------------------------------------------------
-- Table `WEBPORTAL`.`STORAGE_POLICY`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `WEBPORTAL`.`STORAGE_POLICY` ;

CREATE  TABLE IF NOT EXISTS `WEBPORTAL`.`STORAGE_POLICY` (

  `storage` VARCHAR(255) NOT NULL ,
  `attr`    VARCHAR(255) NOT NULL ,
  `value`   TEXT         NOT NULL )

ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `WEBPORTAL`.`FILE_RESTORE_REQUESTS`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `WEBPORTAL`.`FILE_RESTORE_REQUESTS` ;

CREATE  TABLE IF NOT EXISTS `WEBPORTAL`.`FILE_RESTORE_REQUESTS` (

  `exper_id`           INT             NOT NULL ,
  `runnum`             INT             NOT NULL ,
  `file_type`          TEXT            NOT NULL ,

  `irods_filepath`     TEXT            NOT NULL ,
  `irods_src_resource` TEXT            NOT NULL ,
  `irods_dst_resource` TEXT            NOT NULL ,

  `requested_time`     BIGINT UNSIGNED NOT NULL ,
  `requested_uid`      VARCHAR(32)     NOT NULL ,

  `status` ENUM('RECEIVED','SUBMITTED','FAILED','DONE') DEAFULT 'RECEIVED' ,

  CONSTRAINT `FILE_RESTORE_REQUESTS_FK_1`
    FOREIGN KEY (`exper_id` )
    REFERENCES `REGDB`.`EXPERIMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE )

ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `WEBPORTAL`.`MEDIUM_TERM_STORAGE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `WEBPORTAL`.`MEDIUM_TERM_STORAGE` ;

CREATE  TABLE IF NOT EXISTS `WEBPORTAL`.`MEDIUM_TERM_STORAGE` (

  `exper_id`        INT             NOT NULL ,
  `runnum`          INT             NOT NULL ,
  `file_type`       TEXT            NOT NULL ,

  `irods_filepath`  TEXT            NOT NULL ,
  `irods_resource`  TEXT            NOT NULL ,
  `irods_size`      BIGINT UNSIGNED NOT NULL ,

  `registered_time` BIGINT UNSIGNED NOT NULL ,
  `registered_uid`  VARCHAR(32)     NOT NULL ,

  CONSTRAINT `MEDIUM_TERM_STORAGE_FK_1`
    FOREIGN KEY (`exper_id` )
    REFERENCES `REGDB`.`EXPERIMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE )

ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `WEBPORTAL`.`IRODS_CACHE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `WEBPORTAL`.`IRODS_CACHE` ;

CREATE  TABLE IF NOT EXISTS `WEBPORTAL`.`IRODS_CACHE` (

  `id`          INT             NOT NULL AUTO_INCREMENT ,
  `application` VARCHAR(255)    NOT NULL ,
  `create_time` BIGINT UNSIGNED NOT NULL ,

  PRIMARY KEY (`id`) ,
  UNIQUE KEY  (`application`, `create_time`) )

ENGINE = InnoDB
COMMENT = 'Registry for caches';


-- -----------------------------------------------------
-- Table `WEBPORTAL`.`IRODS_TYPE_RUN`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `WEBPORTAL`.`IRODS_TYPE_RUN` ;

CREATE  TABLE IF NOT EXISTS `WEBPORTAL`.`IRODS_TYPE_RUN` (

  `id`        INT         NOT NULL AUTO_INCREMENT ,
  `cache_id`  INT         NOT NULL ,
  `exper_id`  INT         NOT NULL ,
  `file_type` VARCHAR(32) NOT NULL ,
  `runnum`    INT         NOT NULL ,

  PRIMARY KEY (`id`) ,

  INDEX `IRODS_TYPE_RUN_IDX_1` (`cache_id`, `exper_id`, `file_type`, `runnum`) ,

  CONSTRAINT `IRODS_TYPE_RUN_FK_1`
    FOREIGN KEY (`cache_id` )
    REFERENCES `WEBPORTAL`.`IRODS_CACHE` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE ,

  CONSTRAINT `IRODS_TYPE_RUN_FK_2`
    FOREIGN KEY (`exper_id` )
    REFERENCES `REGDB`.`EXPERIMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE )

ENGINE = InnoDB
COMMENT = 'Cache for experiments, file types and runs';


-- -----------------------------------------------------
-- Table `WEBPORTAL`.`IRODS_FILE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `WEBPORTAL`.`IRODS_FILE` ;

CREATE  TABLE IF NOT EXISTS `WEBPORTAL`.`IRODS_FILE` (

  `run_id`   INT         NOT NULL ,
  `run`      INT         NOT NULL ,

  `type`     VARCHAR(32) NOT NULL ,
  `name`     TEXT        NOT NULL ,
  `url`      TEXT        NOT NULL ,

  `checksum` TEXT            NULL ,
  `collName` TEXT            NULL ,
  `ctime`    INT             NULL ,
  `datamode` TEXT            NULL ,
  `id`       INT             NULL ,
  `mtime`    INT             NULL ,
  `owner`    VARCHAR(32)     NULL ,
  `path`     TEXT            NULL ,
  `replStat` TEXT            NULL ,
  `replica`  INT             NULL ,
  `resource` VARCHAR(255)    NULL ,
  `size`     BIGINT UNSIGNED NULL ,

  CONSTRAINT `IRODS_FILE_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `WEBPORTAL`.`IRODS_TYPE_RUN` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE )

ENGINE = InnoDB
COMMENT = 'Cached files';


-- -----------------------------------------------------
-- Table `WEBPORTAL`.`APPLICATION_CONFIG`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `WEBPORTAL`.`APPLICATION_CONFIG` ;

CREATE  TABLE IF NOT EXISTS `WEBPORTAL`.`APPLICATION_CONFIG` (

  `uid`         VARCHAR(32)     NOT NULL ,
  `application` VARCHAR(255)    NOT NULL ,
  `scope`       VARCHAR(255)    NOT NULL ,
  `parameter`   VARCHAR(255)    NOT NULL ,
  `value`       TEXT            NOT NULL ,
  `update_time` BIGINT UNSIGNED NOT NULL ,

  UNIQUE KEY  (`uid`, `application`, `scope`, `parameter`)
)
ENGINE = InnoDB
COMMENT = 'Persistent cache for application parameters' ;



SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
