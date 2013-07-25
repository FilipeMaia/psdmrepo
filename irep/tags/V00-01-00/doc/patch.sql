CREATE  TABLE IF NOT EXISTS `IREP_DEV`.`ISSUE` (

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
    REFERENCES `IREP_DEV`.`EQUIPMENT` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;

CREATE  TABLE IF NOT EXISTS `IREP_DEV`.`ISSUE_ACTION` (

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
    REFERENCES `IREP_DEV`.`ISSUE` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;

CREATE  TABLE IF NOT EXISTS `IREP_DEV`.`ISSUE_ACTION_ATTACHMENT` (

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
    REFERENCES `IREP_DEV`.`ISSUE_ACTION` (`id` )
    ON DELETE CASCADE
    ON UPDATE CASCADE
)
ENGINE = InnoDB;

