SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `LOGBOOK` ;
USE `LOGBOOK`;


-- -----------------------------------------------------
-- Table `LOGBOOK`.`RUN_TABLE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`RUN_TABLE` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`RUN_TABLE` (

  `id`       INT NOT NULL AUTO_INCREMENT ,
  `exper_id` INT NOT NULL ,

  `name`  VARCHAR(255) NOT NULL ,
  `descr` MEDIUMTEXT   NOT NULL ,

  `is_private` ENUM('YES','NO') NOT NULL ,

  `created_uid`  VARCHAR(32)     NOT NULL ,
  `created_time` BIGINT UNSIGNED NOT NULL ,

  `modified_uid`  VARCHAR(32)     NOT NULL ,
  `modified_time` BIGINT UNSIGNED NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE INDEX `RUN_TABLE_IDX_1` (`exper_id` ASC, `name` ASC) ,

  CONSTRAINT `RUN_TABLE_FK_1`
    FOREIGN KEY (`exper_id` )
    REFERENCES `REGDB`.`EXPERIMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB
COMMENT = 'Run tables store various summary informations related to runs';


-- -----------------------------------------------------
-- Table `LOGBOOK`.`RUN_TABLE_COLDEF`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`RUN_TABLE_COLDEF` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`RUN_TABLE_COLDEF` (

  `id`       INT NOT NULL AUTO_INCREMENT ,
  `table_id` INT NOT NULL ,

  `name`   VARCHAR(255) NOT NULL ,
  `type`   VARCHAR(255) NOT NULL ,
  `source` VARCHAR(255) NOT NULL ,

  `is_editable` ENUM('YES','NO') NOT NULL ,

  `position` INT NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE INDEX `RUN_TABLE_COLDEF_IDX_1` (`table_id` ASC, `name` ASC) ,

  CONSTRAINT `RUN_TABLE_COLDEF_FK_1`
    FOREIGN KEY (`table_id` )
    REFERENCES `LOGBOOK`.`RUN_TABLE` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB
COMMENT = 'Column definitions for run tables';


-- -----------------------------------------------------
-- Table `LOGBOOK`.`RUN_TABLE_TEXT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `LOGBOOK`.`RUN_TABLE_TEXT` ;

CREATE  TABLE IF NOT EXISTS `LOGBOOK`.`RUN_TABLE_TEXT` (

  `run_id`    INT NOT NULL ,
  `coldef_id` INT NOT NULL ,

  `val` MEDIUMTEXT NOT NULL ,

  `modified_uid`  VARCHAR(32)     NOT NULL ,
  `modified_time` BIGINT UNSIGNED NOT NULL ,

  PRIMARY KEY (`run_id`,`coldef_id`) ,

  UNIQUE INDEX `RUN_TABLE_TEXT_IDX_1` (`run_id` ASC, `coldef_id` ASC) ,

  CONSTRAINT `RUN_TABLE_TEXT_FK_1`
    FOREIGN KEY (`run_id` )
    REFERENCES `LOGBOOK`.`RUN` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE ,

  CONSTRAINT `RUN_TABLE_TEXT_FK_2`
    FOREIGN KEY (`coldef_id` )
    REFERENCES `LOGBOOK`.`RUN_TABLE_COLDEF` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB
COMMENT = 'Text values for run table cells';



SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
