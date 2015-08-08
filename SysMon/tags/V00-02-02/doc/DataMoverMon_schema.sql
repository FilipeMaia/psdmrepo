SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0;
SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0;
SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='TRADITIONAL';

CREATE SCHEMA IF NOT EXISTS `SYSMON` ;
USE `SYSMON`;

-- -----------------------------------------------------
-- Table `SYSMON`.`dtmv_stat_xfer`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`DTMV_STAT_XFER` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`DTMV_STAT_XFER` (

  `id` INT NOT NULL AUTO_INCREMENT ,

  `direction` ENUM('DSS2FFB', 'FFB2ANA', 'OTHER') NOT NULL ,  -- a direction of the transfer
  `host`      VARCHAR(255)                        NOT NULL ,  -- a machine where the data mover controller runs

  `instr`     VARCHAR(32) ,             -- if available (can extracted from file path or host name)
  `exper_id`  INT ,                     -- if available (can extracted from file path)

  `in_host`  VARCHAR(255) ,             -- set if pulling via a remote host
  `in_path`  VARCHAR(255) NOT NULL ,    -- a local path on the source host

  `out_host` VARCHAR(255) ,             -- set if recording via a remote host
  `out_path` VARCHAR(255) NOT NULL ,    -- a local path on the recording host

  `begin_time` INT UNSIGNED NOT NULL ,  -- recorded when starting the transfer
  `end_time`   INT UNSIGNED ,           -- recorded when finishing the transfer

  `avg_rate` DOUBLE ,                   -- recorded when the `end_time` is set

  PRIMARY KEY (`id`) ,
  CONSTRAINT `DTMV_STAT_XFER_FK_1`
    FOREIGN KEY (`exper_id` )
    REFERENCES `REGDB`.`EXPERIMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE  = InnoDB
COMMENT = 'Data migration transfers';

-- -----------------------------------------------------
-- Table `SYSMON`.`dtmv_stat_rate`
-- -----------------------------------------------------
DROP TABLE IF EXISTS `SYSMON`.`DTMV_STAT_RATE` ;

CREATE  TABLE IF NOT EXISTS `SYSMON`.`DTMV_STAT_RATE` (

  `xfer_id` INT NOT NULL ,

  `relevance_time` INT UNSIGNED  NOT NULL ,  -- when the rate was measured
  `insert_time`    INT UNSIGNED  NOT NULL ,  -- when the rate was reported into the database

  `rate` DOUBLE NOT NULL ,

  CONSTRAINT `DTMV_STAT_RATE_FK_1`
    FOREIGN KEY (`xfer_id` )
    REFERENCES `SYSMON`.`DTMV_STAT_XFER` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE  = InnoDB
COMMENT = 'Actaul measurements for the data migration rates';




SET SQL_MODE=@OLD_SQL_MODE;
SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS;
SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS;

