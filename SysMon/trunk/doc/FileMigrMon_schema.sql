SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `SYSMON` ;
USE `SYSMON`;

-- -----------------------------------------------------
-- Table `SYSMON`.`FM_DELAY_SUBSCRIBER`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`FM_DELAY_SUBSCRIBER` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`FM_DELAY_SUBSCRIBER` (

  `uid` VARCHAR(32) NOT NULL ,                  -- subscribed user's UID

  `subscribed_uid`  VARCHAR(32)     NOT NULL ,  -- a user who made the subscription
  `subscribed_time` BIGINT UNSIGNED NOT NULL ,  -- a time when the subscription was made
  `subscribed_host` VARCHAR(255)    NOT NULL ,  -- a machine where this subscription was made

  `instr`      VARCHAR(32) ,                    -- an instrument to narrow a scope of teh monitoring
  `last_sec`   INT UNSIGNED NOT NULL ,          -- an interval of last seconds to watch for
  `delay_sec`  INT UNSIGNED NOT NULL ,          -- the minimum delay to watch for

  PRIMARY KEY (`uid`)                           -- to prevent multiple subscriptions for the same user
)
ENGINE  = InnoDB
COMMENT = 'Registered subscribers for e-mail notifications for file migration delays';


-- -----------------------------------------------------
-- Table `SYSMON`.`FM_DELAY_EVENT`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`FM_DELAY_EVENT` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`FM_DELAY_EVENT` (

  `name`  VARCHAR(32) NOT NULL ,  -- event name (make it short)
  `descr` TEXT        NOT NULL ,  -- some meaningful description of the event

  PRIMARY KEY (`name`)
)
ENGINE  = InnoDB
COMMENT = 'Specific classes of the file migration delays';

INSERT INTO `SYSMON`.`FM_DELAY_EVENT` VALUES ('xtc.DSS2FFB.begin',  'XTC: DSS to FFB migration has not started') ;
INSERT INTO `SYSMON`.`FM_DELAY_EVENT` VALUES ('xtc.DSS2FFB.end',    'XTC: DSS to FFB migration is taking for too long') ;
INSERT INTO `SYSMON`.`FM_DELAY_EVENT` VALUES ('xtc.FFB2ANA.begin',  'XTC: FFB to ANA migration has not started') ;
INSERT INTO `SYSMON`.`FM_DELAY_EVENT` VALUES ('xtc.FFB2ANA.end',    'XTC: FFB to ANA migration is taking for too long') ;
INSERT INTO `SYSMON`.`FM_DELAY_EVENT` VALUES ('xtc.ANA2HPSS.begin', 'XTC: IRODS registration has not happened') ;
INSERT INTO `SYSMON`.`FM_DELAY_EVENT` VALUES ('xtc.ANA2HPSS.end',   'XTC: Have not been archived to HPSS for too long') ;


-- -----------------------------------------------------
-- Table `SYSMON`.`FM_DELAY_EVENT_SUBSCRIBER`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`FM_DELAY_EVENT_SUBSCRIBER` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`FM_DELAY_EVENT_SUBSCRIBER` (

  `event_name`     VARCHAR(32) NOT NULL ,  -- an event
  `subscriber_uid` VARCHAR(32) NOT NULL ,  -- a subscriber

  UNIQUE KEY (`event_name`,`subscriber_uid`) ,

  CONSTRAINT `FM_DELAY_EVENT_SUBSCRIBER_FK_1`
    FOREIGN KEY (`event_name` )
    REFERENCES `SYSMON`.`FM_DELAY_EVENT` (`name` )
    ON DELETE CASCADE
    ON UPDATE CASCADE ,

  CONSTRAINT `FM_DELAY_EVENT_SUBSCRIBER_FK_2`
    FOREIGN KEY (`subscriber_uid` )
    REFERENCES `SYSMON`.`FM_DELAY_SUBSCRIBER` (`uid` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE  = InnoDB
COMMENT = 'The joint table for subscribing to specific classes of events';



SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

