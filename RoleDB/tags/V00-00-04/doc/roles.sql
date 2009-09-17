CREATE TABLE role (
    id          INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(32) NOT NULL,
    app         VARCHAR(64) NOT NULL,
    INDEX role_name_app_idx (name,app)
);

CREATE TABLE priv (
    id          INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    name        VARCHAR(32) NOT NULL,
    role_id     INT UNSIGNED NOT NULL,
    INDEX role_name_app_idx (name,role_id),
    CONSTRAINT FOREIGN KEY fk_priv_role_id ( role_id ) REFERENCES role ( id ) 
);

CREATE TABLE user (
    id          INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    exp_id      INT,
    user        VARCHAR(32) NOT NULL,
    role_id     INT UNSIGNED NOT NULL NOT NULL,
    INDEX user_exp_idx (exp_id),
    INDEX user_user_idx (user),
    CONSTRAINT FOREIGN KEY fk_priv_role_id ( role_id ) REFERENCES role ( id ) 
);

insert into role (id, name, app) values (1,'Admin','RoleDB');

insert into priv (id, name, role_id) values (1,'create',1);
insert into priv (id, name, role_id) values (2,'delete',1);
insert into priv (id, name, role_id) values (3,'update',1);

insert into user (id, exp_id, user, role_id) values (1,NULL,'salnikov',1);
