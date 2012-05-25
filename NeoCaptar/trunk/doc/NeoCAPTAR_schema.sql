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
  `documentation` TEXT NOT NULL ,
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

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`DICT_ROUTING`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`DICT_ROUTING` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`DICT_ROUTING` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(255) NOT NULL ,
  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid` VARCHAR(32) NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `name` (`name`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`DICT_DEVICE_LOCATION`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`DICT_DEVICE_LOCATION` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`DICT_DEVICE_LOCATION` (
  `id`           INT             NOT NULL AUTO_INCREMENT ,
  `name`         VARCHAR(255)    NOT NULL ,
  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid`  VARCHAR(32)     NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `name` (`name`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`DICT_DEVICE_REGION`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`DICT_DEVICE_REGION` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`DICT_DEVICE_REGION` (
  `id`           INT             NOT NULL AUTO_INCREMENT ,
  `location_id`  INT             NOT NULL ,
  `name`         VARCHAR(255)    NOT NULL ,
  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid`  VARCHAR(32)     NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `name` (`location_id`,`name`) ,
  INDEX `DICT_DEVICE_REGION_FK_1` (`location_id` ASC) ,
  CONSTRAINT `DICT_DEVICE_REGION_FK_1`
    FOREIGN KEY (`location_id` )
    REFERENCES `NEOCAPTAR`.`DICT_DEVICE_LOCATION` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`DICT_DEVICE_COMPONENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`DICT_DEVICE_COMPONENT` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`DICT_DEVICE_COMPONENT` (
  `id`           INT             NOT NULL AUTO_INCREMENT ,
  `region_id`    INT             NOT NULL ,
  `name`         VARCHAR(255)    NOT NULL ,
  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid`  VARCHAR(32)     NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `name` (`region_id`,`name`) ,
  INDEX `DICT_DEVICE_COMPONENT_FK_1` (`region_id` ASC) ,
  CONSTRAINT `DICT_DEVICE_COMPONENT_FK_1`
    FOREIGN KEY (`region_id` )
    REFERENCES `NEOCAPTAR`.`DICT_DEVICE_REGION` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`DICT_INSTR`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`DICT_INSTR` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`DICT_INSTR` (
  `id` INT NOT NULL AUTO_INCREMENT ,
  `name` VARCHAR(255) NOT NULL ,
  `created_time` BIGINT UNSIGNED NOT NULL ,
  `created_uid` VARCHAR(32) NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `name` (`name`))
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`PROJECT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`PROJECT` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`PROJECT` (

  `id`            INT             NOT NULL AUTO_INCREMENT ,
  `owner`         VARCHAR(32)     NOT NULL ,
  `title`         VARCHAR(255)    NOT NULL ,
  `job`           VARCHAR(32)     NOT NULL ,
  `description`   TEXT            NOT NULL ,
  `created_time`  BIGINT UNSIGNED NOT NULL ,
  `due_time`      BIGINT UNSIGNED NOT NULL ,
  `modified_time` BIGINT UNSIGNED NOT NULL ,

  PRIMARY KEY (`id`) ,
  UNIQUE KEY `title` (`title`))

ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`CABLE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`CABLE` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`CABLE` (

  `id`         INT NOT NULL AUTO_INCREMENT ,
  `project_id` INT NOT NULL ,

  `status` ENUM('Planned',
                'Registered',
                'Labeled',
                'Fabrication',
                'Ready',
                'Installed',
                'Commissioned',
                'Damaged',
                'Retired') NOT NULL ,

  `cable`                VARCHAR(32)  NOT NULL ,

  `device`               VARCHAR(255) NOT NULL ,
  `device_location`      VARCHAR(3)   NOT NULL ,
  `device_region`        VARCHAR(3)   NOT NULL ,
  `device_component`     VARCHAR(3)   NOT NULL ,
  `device_counter`       VARCHAR(2)   NOT NULL ,
  `device_suffix`        VARCHAR(3)   NOT NULL ,

  `func`                 VARCHAR(255) NOT NULL ,
  `length`               VARCHAR(32)  NOT NULL ,
  `cable_type`           VARCHAR(32)  NOT NULL ,
  `routing`              VARCHAR(32)  NOT NULL ,

  `origin_name`          VARCHAR(255) NOT NULL ,
  `origin_loc`           VARCHAR(32)  NOT NULL ,
  `origin_rack`          VARCHAR(32)  NOT NULL ,
  `origin_ele`           VARCHAR(32)  NOT NULL ,
  `origin_side`          VARCHAR(32)  NOT NULL ,
  `origin_slot`          VARCHAR(32)  NOT NULL ,
  `origin_conn`          VARCHAR(32)  NOT NULL ,
  `origin_conntype`      VARCHAR(32)  NOT NULL ,
  `origin_pinlist`       VARCHAR(32)  NOT NULL ,
  `origin_station`       VARCHAR(32)  NOT NULL ,
  `origin_instr`         VARCHAR(32)  NOT NULL ,

  `destination_name`     VARCHAR(255) NOT NULL ,
  `destination_loc`      VARCHAR(32)  NOT NULL ,
  `destination_rack`     VARCHAR(32)  NOT NULL ,
  `destination_ele`      VARCHAR(32)  NOT NULL ,
  `destination_side`     VARCHAR(32)  NOT NULL ,
  `destination_slot`     VARCHAR(32)  NOT NULL ,
  `destination_conn`     VARCHAR(32)  NOT NULL ,
  `destination_conntype` VARCHAR(32)  NOT NULL ,
  `destination_pinlist`  VARCHAR(32)  NOT NULL ,
  `destination_station`  VARCHAR(32)  NOT NULL ,
  `destination_instr`    VARCHAR(32)  NOT NULL ,

  PRIMARY KEY (`id`) ,

  INDEX `CABLE_FK_1` (`project_id` ASC) ,
  CONSTRAINT `CABLE_FK_1`
    FOREIGN KEY (`project_id` )
    REFERENCES `NEOCAPTAR`.`PROJECT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;



-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`CABLE_HISTORY`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`CABLE_HISTORY` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`CABLE_HISTORY` (

  `id`         INT             NOT NULL AUTO_INCREMENT ,

  `cable_id`   INT             NOT NULL ,
  `event_time` BIGINT UNSIGNED NOT NULL ,
  `event_uid`  VARCHAR(32)     NOT NULL ,
  `event`      TEXT            NOT NULL ,

  PRIMARY KEY (`id`) ,

  CONSTRAINT `CABLE_HISTORY_FK_1`
    FOREIGN KEY (`cable_id` )
    REFERENCES `NEOCAPTAR`.`CABLE` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`CABLE_HISTORY_COMMENTS`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`CABLE_HISTORY_COMMENTS` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`CABLE_HISTORY_COMMENTS` (

  `cable_history_id` INT  NOT NULL ,
  `comment`          TEXT NOT NULL ,

  CONSTRAINT `CABLE_HISTORY_COMMENTS_FK_1`
    FOREIGN KEY (`cable_history_id` )
    REFERENCES `NEOCAPTAR`.`CABLE_HISTORY` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`PROJECT_HISTORY`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`PROJECT_HISTORY` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`PROJECT_HISTORY` (

  `id`         INT             NOT NULL AUTO_INCREMENT ,

  `project_id` INT             NOT NULL ,
  `event_time` BIGINT UNSIGNED NOT NULL ,
  `event_uid`  VARCHAR(32)     NOT NULL ,
  `event`      TEXT            NOT NULL ,

  PRIMARY KEY (`id`) ,

  CONSTRAINT `PROJECT_HISTORY_FK_1`
    FOREIGN KEY (`project_id` )
    REFERENCES `NEOCAPTAR`.`PROJECT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`PROJECT_HISTORY_COMMENTS`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`PROJECT_HISTORY_COMMENTS` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`PROJECT_HISTORY_COMMENTS` (

  `project_history_id` INT  NOT NULL ,
  `comment`            TEXT NOT NULL ,

  CONSTRAINT `PROJECT_HISTORY_COMMENTS_FK_1`
    FOREIGN KEY (`project_history_id` )
    REFERENCES `NEOCAPTAR`.`PROJECT_HISTORY` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`HISTORY`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`HISTORY` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`HISTORY` (

  `id`         INT             NOT NULL AUTO_INCREMENT ,

  `event_time` BIGINT UNSIGNED NOT NULL ,
  `event_uid`  VARCHAR(32)     NOT NULL ,
  `event`      TEXT            NOT NULL ,

  PRIMARY KEY (`id`)
)
ENGINE = InnoDB;

-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`HISTORY_COMMENTS`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`HISTORY_COMMENTS` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`HISTORY_COMMENTS` (

  `history_id` INT  NOT NULL ,
  `comment`    TEXT NOT NULL ,

  CONSTRAINT `HISTORY_COMMENTS_FK_1`
    FOREIGN KEY (`history_id` )
    REFERENCES `NEOCAPTAR`.`HISTORY` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`CABLENUMBER`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`CABLENUMBER` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`CABLENUMBER` (
  `id`       INT          NOT NULL AUTO_INCREMENT ,
  `location` VARCHAR(32)  NOT NULL ,
  `prefix`   VARCHAR(2)   NOT NULL ,
  `first`    INT UNSIGNED NOT NULL ,
  `last`     INT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `location` (`location`) ,
  UNIQUE KEY `prefix`   (`prefix`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`CABLENUMBER_ALLOCATION`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`CABLENUMBER_ALLOCATION` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`CABLENUMBER_ALLOCATION` (
  `cablenumber_id`   INT             NOT NULL ,
  `cablenumber`      INT UNSIGNED    NOT NULL ,
  `allocated_time`   BIGINT UNSIGNED NOT NULL ,
  `allocated_by_uid` VARCHAR(32)     NOT NULL ,
  UNIQUE KEY `cablenumber_id` (`cablenumber_id`,`cablenumber`) ,
  CONSTRAINT `CABLENUMBER_ALLOCATION_FK_1`
    FOREIGN KEY (`cablenumber_id` )
    REFERENCES `NEOCAPTAR`.`CABLENUMBER` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE)
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`JOBNUMBER`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`JOBNUMBER` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`JOBNUMBER` (
  `id`       INT          NOT NULL AUTO_INCREMENT ,
  `owner`    VARCHAR(32)  NOT NULL ,
  `prefix`   VARCHAR(3)   NOT NULL ,
  `first`    INT UNSIGNED NOT NULL ,
  `last`     INT UNSIGNED NOT NULL ,
  PRIMARY KEY (`id`) ,
  UNIQUE KEY `owner`  (`owner`) ,
  UNIQUE KEY `prefix` (`prefix`))
ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`JOBNUMBER_ALLOCATION`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`JOBNUMBER_ALLOCATION` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`JOBNUMBER_ALLOCATION` (

  `jobnumber_id`     INT             NOT NULL ,
  `project_id`       INT             NOT NULL ,
  `jobnumber`        INT UNSIGNED    NOT NULL ,
  `allocated_time`   BIGINT UNSIGNED NOT NULL ,
  `allocated_by_uid` VARCHAR(32)     NOT NULL ,

  UNIQUE KEY `jobnumber4project` (`jobnumber_id`,`project_id`) ,

  CONSTRAINT `JOBNUMBER_ALLOCATION_FK_1`
    FOREIGN KEY (`jobnumber_id` )
    REFERENCES `NEOCAPTAR`.`JOBNUMBER` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE ,

  CONSTRAINT `JOBNUMBER_ALLOCATION_FK_2`
    FOREIGN KEY (`project_id` )
    REFERENCES `NEOCAPTAR`.`PROJECT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE )

ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`USER`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`USER` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`USER` (

  `uid`  VARCHAR(32)                 NOT NULL ,

  `role` ENUM('ADMINISTRATOR','PROJMANAGER','OTHER') NOT NULL ,

  `name` VARCHAR(255)                NOT NULL ,
  `added_time` BIGINT UNSIGNED       NOT NULL ,
  `added_uid` VARCHAR(32)            NOT NULL ,
  `last_active_time` BIGINT UNSIGNED NULL ,

  PRIMARY KEY (`uid`))

ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`NOTIFY_EVENT_TYPE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` (

  `id`          INT NOT NULL AUTO_INCREMENT ,

  `recipient`   ENUM('ADMINISTRATOR','PROJMANAGER','OTHER') NOT NULL ,
  `name`        VARCHAR(255)                                NOT NULL ,
  `scope`       ENUM('PROJECT','CABLE')                     NOT NULL ,
  `description` TEXT                                        NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE KEY `name` (`recipient`,`name`,`scope`) )

ENGINE = InnoDB;

INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'ADMINISTRATOR','on_project_create',  'PROJECT','New project created');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'ADMINISTRATOR','on_project_assign',  'PROJECT','Project assigned to a different owner');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'ADMINISTRATOR','on_project_delete',  'PROJECT','Project deleted');


INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_project_create',  'PROJECT','New project created and assigned to me');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_project_assign',  'PROJECT','Someone else\'s project assigned to me');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_project_deassign','PROJECT','My project assigned to someone else');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_project_delete',  'PROJECT','My project deleted');

INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_cable_create',    'CABLE',  'New cable added to my project');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_cable_delete',    'CABLE',  'Cable deleted');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_cable_edit',      'CABLE',  'Cable edited');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_register',        'CABLE',  'Cable registered with an official #');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_label',           'CABLE',  'Cable label finalized and locked');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_fabrication',     'CABLE',  'Cable fabrication requested');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_ready',           'CABLE',  'Cable is ready to be installed');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_install',         'CABLE',  'Cable is installed');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_commission',      'CABLE',  'Cable is commissioned');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_damage',          'CABLE',  'Cable is damaged');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'PROJMANAGER',  'on_retire',          'CABLE',  'Cable is retired');

INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'OTHER',        'on_project_create',  'PROJECT','New project created');

INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'OTHER',        'on_fabrication',     'CABLE',  'Cable fabrication ordered');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'OTHER',        'on_ready',           'CABLE',  'Cable is ready to be installed');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'OTHER',        'on_install',         'CABLE',  'Cable is installed');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'OTHER',        'on_commission',      'CABLE',  'Cable is commissioned');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'OTHER',        'on_damage',          'CABLE',  'Cable is damaged');
INSERT INTO `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` VALUES(NULL,'OTHER',        'on_retire',          'CABLE',  'Cable is retired');



-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`NOTIFY`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`NOTIFY` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`NOTIFY` (

    `id`            INT              NOT NULL AUTO_INCREMENT ,
    `uid`           VARCHAR(32)      NOT NULL ,
    `event_type_id` INT              NOT NULL ,
    `enabled`       ENUM('ON','OFF') NOT NULL ,

  PRIMARY KEY (`id`) ,

  UNIQUE KEY `event_type` (`uid`,`event_type_id`) ,

  CONSTRAINT `NOTIFY_FK_1`
    FOREIGN KEY (`event_type_id` )
    REFERENCES `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)

ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`NOTIFY_SCHEDULE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`NOTIFY_SCHEDULE` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`NOTIFY_SCHEDULE` (

    `recipient` ENUM('ADMINISTRATOR','PROJMANAGER','OTHER') NOT NULL ,
    `mode`      ENUM('INSTANT','DELAYED')                   NOT NULL ,

  UNIQUE KEY `recipient` (`recipient`,`mode`))

ENGINE = InnoDB;

INSERT INTO `NEOCAPTAR`.`NOTIFY_SCHEDULE` VALUES('ADMINISTRATOR','INSTANT');
INSERT INTO `NEOCAPTAR`.`NOTIFY_SCHEDULE` VALUES('PROJMANAGER','INSTANT');
INSERT INTO `NEOCAPTAR`.`NOTIFY_SCHEDULE` VALUES('OTHER','DELAYED');


-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`NOTIFY_QUEUE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`NOTIFY_QUEUE` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`NOTIFY_QUEUE` (

    `id` INT NOT NULL AUTO_INCREMENT ,

    `event_type_id`        INT             NOT NULL ,
    `event_time`           BIGINT UNSIGNED NOT NULL ,
    `event_originator_uid` VARCHAR(32)     NOT NULL ,
    `recipient_uid`        VARCHAR(32)     NOT NULL ,

  PRIMARY KEY (`id`) ,

  CONSTRAINT `NOTIFY_QUEUE_FK_1`
    FOREIGN KEY (`event_type_id` )
    REFERENCES `NEOCAPTAR`.`NOTIFY_EVENT_TYPE` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE)

ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`NOTIFY_QUEUE_PROJECT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`NOTIFY_QUEUE_PROJECT` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`NOTIFY_QUEUE_PROJECT` (

    `notify_queue_id` INT  NOT NULL ,

    `project_id`        INT             NULL ,  -- NOTE: this isn't a foreign key because
                                                -- we want this entry to stay in the table even
                                                -- when the project gets deleted.
    `project_title`     TEXT            NULL ,
    `project_owner_uid` VARCHAR(32)     NULL ,
    `project_due_time`  BIGINT UNSIGNED NULL ,

  CONSTRAINT `NOTIFY_QUEUE_PROJECT_FK_1`
    FOREIGN KEY (`notify_queue_id` )
    REFERENCES `NEOCAPTAR`.`NOTIFY_QUEUE` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE )

ENGINE = InnoDB;


-- -----------------------------------------------------
-- Table `NEOCAPTAR`.`NOTIFY_QUEUE_CABLE`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `NEOCAPTAR`.`NOTIFY_QUEUE_CABLE` ;

CREATE  TABLE IF NOT EXISTS `NEOCAPTAR`.`NOTIFY_QUEUE_CABLE` (

    `notify_queue_id` INT NOT NULL ,
    `cable_id`        INT     NULL ,  -- NOTE: this isn't a foreign key because
                                      -- we want this entry to stay in the table even
                                      -- when the cable gets deleted.
    `cable_info`      TEXT    NULL ,

  CONSTRAINT `NOTIFY_QUEUE_CABLE_FK_1`
    FOREIGN KEY (`notify_queue_id` )
    REFERENCES `NEOCAPTAR`.`NOTIFY_QUEUE` (`id`)
    ON DELETE CASCADE
    ON UPDATE CASCADE )

ENGINE = InnoDB;


SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;
