<?php

namespace FileMgr;

require_once('filemgr.inc.php');

use FileMgr\FileMgrException;

/*
 * The utility class for database connections.
 */
class DbConnection {

    /* Error codes from MySQL documentation. See full list at:
     *
     *   http://dev.mysql.com/doc/refman/5.0/en/error-messages-server.html
     */
    public static $ER_DUP_ENTRY = 1062;  // Message: Duplicate entry '%s' for key %d

    /*
     * ================
     *   DATA MEMBERS
     * ================
     */

    /* Parameters of the object
     */
    public $host;
    public $user;
    public $password;
    public $database;

    /* Current state of the object
     */
    private $link;
    private $in_transaction = false;

    /*
     * =======================================
     *   OBJECT CONSTRUCTION AND DESTRUCTION
     * =======================================
     */

    /**
     * Constructor
     *
     * The constructor won't make any actual connection attempts. This will be deffered
     * to operations dealing with queries, transactions, etc.
     *
     * @param string $host
     * @param string $user
     * @param string $password
     * @param string $database 
     */
    public function __construct ($host, $user, $password, $database) {
        $this->host      = $host;
        $this->user      = $user;
        $this->password  = $password;
        $this->database  = $database;
    }

    /**
     * Destructor
     */
    public function __destruct () {
    	// Do not close this connection from here because it might be shared
    	// with other instances of the class (also from other APIs).
    	//
        //if (isset($this->link)) mysql_close($this->link);
    }

    /*
     * ================================
     *   MySQL TRANSACTION MANAGEMENT
     * ================================
     */
    public function begin () {
        $this->connect();
        if ($this->in_transaction) return;
        $this->transaction('BEGIN');
        $this->in_transaction = true;
    }

    public function commit () {
        $this->connect();
        if (!$this->in_transaction) return;
        $this->transaction('COMMIT');
        $this->in_transaction = false;
    }

    public function rollback () {
        $this->connect();
        if (!$this->in_transaction) return;
        $this->transaction('ROLLBACK');
        $this->in_transaction = false;
    }

    private function transaction ($transition) {
        if (!mysql_query($transition, $this->link))
            throw new FileMgrException (
                __METHOD__,
                "MySQL error: ".mysql_error($this->link).', in query: '.$transition,
                $this->errno());
    }

    /* =================
     *   MySQL QUERIES
     * =================
     */
    public function query ($sql) {

        $this->connect();

        if (!$this->in_transaction)
            throw new FileMgrException (
                __METHOD__, "no active transaction");

        $result = mysql_query($sql, $this->link);
        if (!$result)
            throw new FileMgrException (
                __METHOD__,
                "MySQL error: ".mysql_error($this->link).', in query: '.$sql,
                $this->errno());
        return $result;
    }

    /* ==========================================================
     *   MISC. OPERATIONS REQUIREING DATABASE SERVER CONNECTION
     * ==========================================================
     */
    public function escape_string ($text) {
        $this->connect();
        return mysql_real_escape_string($text, $this->link);  }

    public function errno() { return mysql_errno( $this->link ); }

    /**
     * Make a connection if this hasn't been done yet.
     */
    protected function connect () {

        if (!isset($this->link)) {

            /* Connect to MySQL server and register the shutdown function
             * to clean up after self by doing 'ROLLBACK'. This would allow
             * to use so called 'persistent' MySQL connections.
             * 
             * NOTE: using the 'persistent' connection. This connection won't be
             * closed by 'mysql_close()'.
             */
            $new_link = false;

            $this->link = mysql_pconnect($this->host, $this->user, $this->password, $new_link);
            if (!$this->link)
                throw new FileMgrException (
                    __METHOD__,
                    "MySQL error: ".mysql_error($this->link).", in function: mysql_connect",
                    $this->errno());

            if (!mysql_select_db($this->database, $this->link))
                throw new FileMgrException (
                    __METHOD__,
                    "MySQL error: ".mysql_error($this->link).", in function: mysql_select_db",
                    $this->errno());

            $sql = "SET SESSION SQL_MODE='ANSI'";
            if (!mysql_query($sql, $this->link))
                throw new FileMgrException (
                    __METHOD__,
                    "MySQL error: ".mysql_error($this->link).', in query: '.$sql,
                    $this->errno());

            $sql = "SET SESSION AUTOCOMMIT=0";
            if (!mysql_query($sql, $this->link))
                throw new FileMgrException (
                    __METHOD__,
                    "MySQL error: ".mysql_error($this->link).', in query: '.$sql,
                    $this->errno());

            register_shutdown_function(array($this, "rollback"));
        }
    }
}
?>
