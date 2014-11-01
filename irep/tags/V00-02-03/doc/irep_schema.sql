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
-- Table `IREP`.`DICT_MODEL_ATTACHMENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`DICT_MODEL_ATTACHMENT` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`DICT_MODEL_ATTACHMENT` (

  `id`       INT NOT NULL AUTO_INCREMENT ,
  `model_id` INT NOT NULL ,

  `rank`     INT NOT NULL ,

  `name`             TEXT            NOT NULL ,
  `document_type`    VARCHAR(255)    NOT NULL ,
  `document_size`    BIGINT UNSIGNED NOT NULL ,
  `document`         LONGBLOB        NOT NULL ,
  `document_preview` LONGBLOB ,

  `create_time`   BIGINT UNSIGNED NOT NULL ,
  `create_uid`    VARCHAR(32)     NOT NULL ,

  PRIMARY KEY (`id`) ,

  CONSTRAINT `DICT_MODEL_ATTACHMENT_FK_1`
    FOREIGN KEY (`model_id` )
    REFERENCES `IREP`.`DICT_MODEL` (`id` )
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
-- Table `IREP`.`DICT_ROOM`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`DICT_ROOM` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`DICT_ROOM` (

  `id`           INT             NOT NULL AUTO_INCREMENT ,
  `location_id`  INT             NOT NULL ,

  `name`         VARCHAR(255)    NOT NULL ,
  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid`  VARCHAR(32)     NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE KEY `name` (`name`,`location_id`) ,

  CONSTRAINT `DICT_ROOM_FK_1`
    FOREIGN KEY (`location_id` )
    REFERENCES `IREP`.`DICT_LOCATION` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;



-- -----------------------------------------------------
-- Table `IREP`.`DICT_STATUS`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`DICT_STATUS` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`DICT_STATUS` (

  `id`            INT              NOT NULL AUTO_INCREMENT ,
  `name`          VARCHAR(255)     NOT NULL ,

  `is_locked`     ENUM('YES','NO') NOT NULL ,

  `create_time`   BIGINT UNSIGNED  NOT NULL ,
  `create_uid`    VARCHAR(32)      NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE KEY `name` (`name`)

)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IREP`.`DICT_STATUS2`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`DICT_STATUS2` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`DICT_STATUS2` (

  `id`            INT              NOT NULL AUTO_INCREMENT ,
  `status_id`     INT              NOT NULL ,

  `name`          VARCHAR(255)     NOT NULL ,
  `is_locked`     ENUM('YES','NO') NOT NULL ,

  `create_time`   BIGINT UNSIGNED  NOT NULL ,
  `create_uid`    VARCHAR(32)      NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE KEY `name` (`status_id`,`name`) ,


  CONSTRAINT `DICT_STATUS2_FK_1`
    FOREIGN KEY (`status_id` )
    REFERENCES `IREP`.`DICT_STATUS` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;


INSERT INTO IREP.DICT_STATUS VALUES(NULL,'Unknown','YES',0,'gapon') ;
INSERT INTO IREP.DICT_STATUS2 VALUES(NULL,(SELECT LAST_INSERT_ID()),'','YES',0,'gapon') ;
INSERT INTO IREP.DICT_STATUS VALUES(NULL,'Good','YES',0,'gapon') ;
INSERT INTO IREP.DICT_STATUS2 VALUES(NULL,(SELECT LAST_INSERT_ID()),'','YES',0,'gapon') ;
INSERT INTO IREP.DICT_STATUS VALUES(NULL,'Bad','YES',0,'gapon') ;
INSERT INTO IREP.DICT_STATUS2 VALUES(NULL,(SELECT LAST_INSERT_ID()),'','YES',0,'gapon') ;
INSERT INTO IREP.DICT_STATUS VALUES(NULL,'Salvaged','YES',0,'gapon') ;
INSERT INTO IREP.DICT_STATUS2 VALUES(NULL,(SELECT LAST_INSERT_ID()),'','YES',0,'gapon') ;
INSERT INTO IREP.DICT_STATUS VALUES(NULL,'Lost','YES',0,'gapon') ;
INSERT INTO IREP.DICT_STATUS2 VALUES(NULL,(SELECT LAST_INSERT_ID()),'','YES',0,'gapon') ;


-- -----------------------------------------------------
-- Table `IREP`.`EQUIPMENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`EQUIPMENT` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`EQUIPMENT` (

  `id`        INT NOT NULL AUTO_INCREMENT ,
  `parent_id` INT ,

  `status`  VARCHAR(255) NOT NULL ,
  `status2` VARCHAR(255) NOT NULL ,

  `manufacturer` VARCHAR(255) NOT NULL ,
  `model`        VARCHAR(255) NOT NULL ,
  `serial`       VARCHAR(255) NOT NULL ,

  `description` TEXT NOT NULL ,

  `slacid`        INT UNSIGNED NOT NULL ,
  `pc_num`        CHAR(7)      NOT NULL ,
  `location`      VARCHAR(255) NOT NULL ,
  `room`          VARCHAR(255) NOT NULL ,
  `rack`          VARCHAR(255) NOT NULL ,
  `elevation`     VARCHAR(255) NOT NULL ,
  `custodian_uid` VARCHAR(32)  NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE KEY `slacid` (`slacid`) ,

  CONSTRAINT `EQUIPMENT_FK_1`
    FOREIGN KEY (`parent_id` )
    REFERENCES `IREP`.`EQUIPMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB ;


-- -----------------------------------------------------
-- Table `IREP`.`EQUIPMENT_ATTACHMENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`EQUIPMENT_ATTACHMENT` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`EQUIPMENT_ATTACHMENT` (

  `id`           INT             NOT NULL AUTO_INCREMENT ,
  `equipment_id` INT             NOT NULL ,

  `name`             TEXT            NOT NULL ,
  `document_type`    VARCHAR(255)    NOT NULL ,
  `document_size`    BIGINT UNSIGNED NOT NULL ,
  `document`         LONGBLOB        NOT NULL ,
  `document_preview` LONGBLOB ,

  `create_time`   BIGINT UNSIGNED NOT NULL ,
  `create_uid`    VARCHAR(32)     NOT NULL ,

  PRIMARY KEY (`id`) ,

  CONSTRAINT `EQUIPMENT_ATTACHMENT_FK_1`
    FOREIGN KEY (`equipment_id` )
    REFERENCES `IREP`.`EQUIPMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IREP`.`EQUIPMENT_TAG`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`EQUIPMENT_TAG` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`EQUIPMENT_TAG` (

  `id`           INT             NOT NULL AUTO_INCREMENT ,
  `equipment_id` INT             NOT NULL ,

  `name`         TEXT            NOT NULL ,

  `create_time`  BIGINT UNSIGNED NOT NULL ,
  `create_uid`   VARCHAR(32)     NOT NULL ,

  PRIMARY KEY (`id`) ,

  CONSTRAINT `EQUIPMENT_TAG_FK_1`
    FOREIGN KEY (`equipment_id` )
    REFERENCES `IREP`.`EQUIPMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;


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
-- Table `IREP`.`ISSUE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`ISSUE` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`ISSUE` (

  `id`           INT             NOT NULL AUTO_INCREMENT ,
  `equipment_id` INT             NOT NULL ,

  `description`  TEXT            NOT NULL ,
  `category`     VARCHAR(255)    NOT NULL ,

  `open_time`    BIGINT UNSIGNED NOT NULL ,
  `open_uid`     VARCHAR(32)     NOT NULL ,

  `resolve_time` BIGINT UNSIGNED NOT NULL ,
  `resolve_uid`  VARCHAR(32)     NOT NULL ,

  PRIMARY KEY (`id`) ,

  CONSTRAINT `ISSUE_FK_1`
    FOREIGN KEY (`equipment_id` )
    REFERENCES `IREP`.`EQUIPMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IREP`.`ISSUE_ACTION`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`ISSUE_ACTION` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`ISSUE_ACTION` (

  `id`           INT               NOT NULL AUTO_INCREMENT ,
  `issue_id`     INT               NOT NULL ,

  `description`  TEXT              NOT NULL ,
  `action`       ENUM( 'OPEN' ,
                       'COMMENT' ,
                       'RE-OPEN' ,
                       'RESOLVE')  NOT NULL ,

  `action_time`  BIGINT UNSIGNED   NOT NULL ,
  `action_uid`   VARCHAR(32)       NOT NULL ,

  PRIMARY KEY (`id`) ,

  CONSTRAINT `ISSUE_ACTION_FK_1`
    FOREIGN KEY (`issue_id` )
    REFERENCES `IREP`.`ISSUE` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IREP`.`ISSUE_ACTION_ATTACHMENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`ISSUE_ACTION_ATTACHMENT` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`ISSUE_ACTION_ATTACHMENT` (

  `id`           INT                 NOT NULL AUTO_INCREMENT ,
  `action_id`    INT                 NOT NULL ,

  `name`             TEXT            NOT NULL ,
  `document_type`    VARCHAR(255)    NOT NULL ,
  `document_size`    BIGINT UNSIGNED NOT NULL ,
  `document`         LONGBLOB        NOT NULL ,
  `document_preview` LONGBLOB ,

  `create_time`   BIGINT UNSIGNED NOT NULL ,
  `create_uid`    VARCHAR(32)     NOT NULL ,

  PRIMARY KEY (`id`) ,

  CONSTRAINT `ISSUE_ACTION_ATTACHMENT_FK_1`
    FOREIGN KEY (`action_id` )
    REFERENCES `IREP`.`ISSUE_ACTION` (`id` )
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
-- Table `IREP`.`SLACID_RANGE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`SLACID_RANGE` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`SLACID_RANGE` (

  `id`    INT          NOT NULL AUTO_INCREMENT ,
  `first` INT UNSIGNED NOT NULL ,
  `last`  INT UNSIGNED NOT NULL ,

  `description` TEXT NOT NULL ,

  PRIMARY KEY (`id`)
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `IREP`.`SLACID_ALLOCATED`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `IREP`.`SLACID_ALLOCATED` ;

CREATE  TABLE IF NOT EXISTS `IREP`.`SLACID_ALLOCATED` (

  `range_id`         INT             NOT NULL ,
  `slacid`           INT UNSIGNED    NOT NULL ,
  `equipment_id`     INT             NOT NULL ,
  `allocated_time`   BIGINT UNSIGNED NOT NULL ,
  `allocated_by_uid` VARCHAR(32)     NOT NULL ,

  UNIQUE KEY `slacid` (`range_id`,`slacid`,`equipment_id`) ,

  CONSTRAINT `SLACID_ALLOCATED_FK_1`
    FOREIGN KEY (`range_id` )
    REFERENCES `IREP`.`SLACID_RANGE` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE ,

  CONSTRAINT `SLACID_ALLOCATED_FK_2`
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
