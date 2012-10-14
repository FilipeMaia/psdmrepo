SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 ;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 ;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL' ;

CREATE SCHEMA IF NOT EXISTS `IREP` ;
USE `IREP` ;


-- -----------------------------------------------------
-- Table `IREP`.`DICT_MANUFACTURER`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`DICT_MANUFACTURER` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`DICT_MANUFACTURER` (

  `id` INT NOT NULL AUTO_INCREMENT ,

  `name` VARCHAR(255) NOT NULL ,
  `url`  TEXT         NOT NULL ,

  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid`  VARCHAR(32)     NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE KEY `name` (`name`)
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IREP`.`DICT_MODEL`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`DICT_MODEL` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`DICT_MODEL` (

  `id`              INT NOT NULL AUTO_INCREMENT ,
  `manufacturer_id` INT NOT NULL ,

  `name` VARCHAR(255) NOT NULL ,
  `url`  TEXT         NOT NULL ,

  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid`  VARCHAR(32)     NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE KEY `name` (`manufacturer_id`,`name`) ,

  INDEX `DICT_MODEL_FK_1` (`manufacturer_id` ASC) ,

  CONSTRAINT `DICT_MODEL_FK_1`
    FOREIGN KEY (`manufacturer_id` )
    REFERENCES `IREP`.`DICT_MANUFACTURER` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IREP`.`DICT_LOCATION`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`DICT_LOCATION` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`DICT_LOCATION` (

  `id`           INT             NOT NULL AUTO_INCREMENT ,
  `name`         VARCHAR(255)    NOT NULL ,
  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid`  VARCHAR(32)     NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE KEY `name` (`name`)
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IREP`.`EQUIPMENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`EQUIPMENT` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`EQUIPMENT` (

  `id` INT NOT NULL AUTO_INCREMENT ,

  `status` ENUM('Unknown') NOT NULL ,

  `equipment`    VARCHAR(255) NOT NULL ,
  `manufacturer` VARCHAR(255) NOT NULL ,
  `model`        VARCHAR(255) NOT NULL ,

  `description` TEXT NOT NULL ,

  `slacid`        INT UNSIGNED NOT NULL ,
  `pc_num`        CHAR(7)      NOT NULL ,
  `location`      VARCHAR(255) NOT NULL ,
  `custodian_uid` VARCHAR(32)  NOT NULL ,

  PRIMARY KEY (`id`)
)
ENGINE = InnoDB ;


-- -----------------------------------------------------
-- Table `IREP`.`EQUIPMENT_HISTORY`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`EQUIPMENT_HISTORY` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`EQUIPMENT_HISTORY` (

  `id`           INT             NOT NULL AUTO_INCREMENT ,
  `equipment_id` INT             NOT NULL ,
  `event_time`   BIGINT UNSIGNED NOT NULL ,
  `event_uid`    VARCHAR(32)     NOT NULL ,
  `event`        TEXT            NOT NULL ,

  PRIMARY KEY (`id`) ,

  CONSTRAINT `EQUIPMENT_HISTORY_FK_1`
    FOREIGN KEY (`equipment_id` )
    REFERENCES `IREP`.`EQUIPMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IREP`.`EQUIPMENT_HISTORY_COMMENTS`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`EQUIPMENT_HISTORY_COMMENTS` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`EQUIPMENT_HISTORY_COMMENTS` (

  `equipment_history_id` INT  NOT NULL ,
  `comment`              TEXT NOT NULL ,

  CONSTRAINT `EQUIPMENT_HISTORY_COMMENTS_FK_1`
    FOREIGN KEY (`equipment_history_id` )
    REFERENCES `IREP`.`EQUIPMENT_HISTORY` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;


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


-- -----------------------------------------------------
-- Table `IREP`.`SLACIDNUMBER_RANGE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`SLACIDNUMBER_RANGE` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`SLACIDNUMBER_RANGE` (

  `id`    INT          NOT NULL AUTO_INCREMENT ,
  `first` INT UNSIGNED NOT NULL ,
  `last`  INT UNSIGNED NOT NULL ,

  PRIMARY KEY (`id`)
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IREP`.`SLACIDNUMBER_ALLOCATED`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`SLACIDNUMBER_ALLOCATED` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`SLACIDNUMBER_ALLOCATED` (

  `range_id`         INT             NOT NULL ,
  `slacidnumber`     INT UNSIGNED    NOT NULL ,
  `equipment_id`     INT             NOT NULL ,
  `allocated_time`   BIGINT UNSIGNED NOT NULL ,
  `allocated_by_uid` VARCHAR(32)     NOT NULL ,

  UNIQUE KEY `slacidnumber` (`range_id`,`slacidnumber`,`equipment_id`) ,

  CONSTRAINT `SLACIDNUMBER_ALLOCATED_FK_1`
    FOREIGN KEY (`range_id` )
    REFERENCES `IREP`.`SLACIDNUMBER_RANGE` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE ,

  CONSTRAINT `SLACIDNUMBER_ALLOCATED_FK_2`
    FOREIGN KEY (`equipment_id` )
    REFERENCES `IREP`.`EQUIPMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IREP`.`NOTIFY_EVENT_TYPE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`NOTIFY_EVENT_TYPE` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`NOTIFY_EVENT_TYPE` (

  `id` INT NOT NULL AUTO_INCREMENT ,

  `recipient` ENUM('ADMINISTRATOR','EDITOR','OTHER') NOT NULL ,

  `name`        VARCHAR(255)      NOT NULL ,
  `scope`       ENUM('EQUIPMENT') NOT NULL ,
  `description` TEXT              NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE KEY `name` (`recipient`,`name`,`scope`)
)
ENGINE = InnoDB;

INSERT INTO `IREP`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'ADMINISTRATOR','on_equipment_add', 'EQUIPMENT','New equipment added to inventory');
INSERT INTO `IREP`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'EDITOR',       'on_equipment_add', 'EQUIPMENT','New equipment added to inventory');
INSERT INTO `IREP`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'OTHER',        'on_equipment_add', 'EQUIPMENT','New equipment added to inventory');


-- -----------------------------------------------------
-- Table `IREP`.`NOTIFY`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`NOTIFY` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`NOTIFY` (

    `id`            INT              NOT NULL AUTO_INCREMENT ,
    `uid`           VARCHAR(32)      NOT NULL ,
    `event_type_id` INT              NOT NULL ,
    `enabled`       ENUM('ON','OFF') NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE KEY `event_type` (`uid`,`event_type_id`) ,

  CONSTRAINT `NOTIFY_FK_1`
    FOREIGN KEY (`event_type_id` )
    REFERENCES `IREP`.`NOTIFY_EVENT_TYPE` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IREP`.`NOTIFY_SCHEDULE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`NOTIFY_SCHEDULE` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`NOTIFY_SCHEDULE` (

    `recipient` ENUM('ADMINISTRATOR','EDITOR','OTHER') NOT NULL ,
    `mode`      ENUM('INSTANT','DELAYED')              NOT NULL ,

  UNIQUE KEY `recipient` (`recipient`,`mode`)
)
ENGINE = InnoDB;

INSERT INTO `IREP`.`NOTIFY_SCHEDULE` VALUES('ADMINISTRATOR','INSTANT');
INSERT INTO `IREP`.`NOTIFY_SCHEDULE` VALUES('EDITOR',       'INSTANT');
INSERT INTO `IREP`.`NOTIFY_SCHEDULE` VALUES('OTHER',        'DELAYED');


-- -----------------------------------------------------
-- Table `IREP`.`NOTIFY_QUEUE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`NOTIFY_QUEUE` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`NOTIFY_QUEUE` (

    `id`                   INT             NOT NULL AUTO_INCREMENT ,
    `event_type_id`        INT             NOT NULL ,
    `event_time`           BIGINT UNSIGNED NOT NULL ,
    `event_originator_uid` VARCHAR(32)     NOT NULL ,
    `recipient_uid`        VARCHAR(32)     NOT NULL ,

  PRIMARY KEY (`id`) ,

  CONSTRAINT `NOTIFY_QUEUE_FK_1`
    FOREIGN KEY (`event_type_id` )
    REFERENCES `IREP`.`NOTIFY_EVENT_TYPE` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IREP`.`NOTIFY_QUEUE_EQUIPMENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`NOTIFY_QUEUE_EQUIPMENT` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`NOTIFY_QUEUE_EQUIPMENT` (

    `notify_queue_id` INT NOT NULL ,
    `equipment_id`    INT     NULL ,  -- NOTE: this isn't a foreign key because
                                      -- we want this entry to stay in the table even
                                      -- when the equipment gets deleted.

    `equipment_info` TEXT NULL ,

  CONSTRAINT `NOTIFY_QUEUE_EQUIPMENT_FK_1`
    FOREIGN KEY (`notify_queue_id` )
    REFERENCES `IREP`.`NOTIFY_QUEUE` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;









SET SQL_MODE=@OLD_SQL_MODE ;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS ;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS ;
