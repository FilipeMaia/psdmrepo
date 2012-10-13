SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 ;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 ;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL' ;

CREATE SCHEMA IF NOT EXISTS `IREP` ;
USE `IREP` ;

-- -----------------------------------------------------
-- Table `IREP`.`USER`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`USER` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`USER` (

  `uid` VARCHAR(32) NOT NULL ,

  `role` ENUM('ADMINISTRATOR','EDITOR','OTHER') NOT NULL ,

  `name`             VARCHAR(255)    NOT NULL ,
  `added_time`       BIGINT UNSIGNED NOT NULL ,
  `added_uid`        VARCHAR(32)     NOT NULL ,
  `last_active_time` BIGINT UNSIGNED     NULL ,

  PRIMARY KEY (`uid`)
)
ENGINE = InnoDB ;

-- -----------------------------------------------------
-- Table `IREP`.`USER_PRIV`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`USER_PRIV` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`USER_PRIV` (

  `uid` VARCHAR(32) NOT NULL ,

  `dict_priv` ENUM('YES','NO') NOT NULL ,

  CONSTRAINT `USER_PRIV_FK_1`
    FOREIGN KEY (`uid` )
    REFERENCES `IREP`.`USER` (`uid` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB ;

INSERT INTO `IREP`.`USER` VALUES ('gapon','ADMINISTRATOR','Igor Gaponenko',0,'gapon',NULL) ;
INSERT INTO `IREP`.`USER_PRIV` VALUES ('gapon', 'YES') ;


SET SQL_MODE=@OLD_SQL_MODE ;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS ;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS ;
