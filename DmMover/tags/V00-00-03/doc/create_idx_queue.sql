--
-- Create tables for Index file (Idx) creation tools
-- This is a sqlite3 database
--

PRAGMA foreign_keys = on;

create table StatusEnum (
       id INTEGER PRIMARY KEY AUTOINCREMENT,
       statusName TEXT UNIQUE
);
INSERT INTO StatusEnum (statusName) VALUES ('NEW');
INSERT INTO StatusEnum (statusName) VALUES ('SUBMIT');
INSERT INTO StatusEnum (statusName) VALUES ('ERROR');
INSERT INTO StatusEnum (statusName) VALUES ('FAIL');
INSERT INTO StatusEnum (statusName) VALUES ('DONE');

create table idx( 
    fn TEXT PRIMARY KEY,
    fpath TEXT NOT NULL,
    date_added INTEGER,
    date_status INTEGER,
    status REFERENCES StatusEnum(statusName) DEFAULT 'NEW'
);



