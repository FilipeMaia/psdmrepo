<?php

namespace AuthDB;

require_once( 'authdb.inc.php' );
require_once( 'lusitime/lusitime.inc.php' );

use LusiTime\LusiTime;

/**
 * Class Logger encapsulates operations with the logging database
 */
class Logger {

    // ---------------------------------------------------
    // --- SIMPLIFIED INTERFACE AND ITS IMPLEMENTATION ---
    // ---------------------------------------------------

    private static $instance = null;

    /**
     * Singleton to simplify certain operations.
     *
     * @return unknown_type
     */
    public static function instance() {
        if( is_null( Logger::$instance )) Logger::$instance =
            new Logger (
                LOGGER_DEFAULT_HOST,
                LOGGER_DEFAULT_USER,
                LOGGER_DEFAULT_PASSWORD,
                LOGGER_DEFAULT_DATABASE );
        return Logger::$instance;
    }
    
    /* Parameters of the object
     */
    private $host;
    private $user;
    private $password;
    public  $database;

    /* Current state of the object
     */
    private $link;
    private $in_transaction = false;

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
    public function __construct ( $host, $user, $password, $database ) {
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
        //if( isset( $this->link )) mysql_close( $this->link );
    }

    /*
     * ================================
     *   METHODS FOR REPORTING EVENTS
     * ================================
     */
    public function group_management ( $operation, $user_name, $group_name, $group_type = 'POSIX' ) {
        $this->connect();
        $authdb = AuthDB::instance();
        $authdb->begin();
        $now = LusiTime::now();
        $sql =
            "INSERT INTO group_management VALUES (NULL,".$now->to64().
            ",'".$this->escape_string( $authdb->authName()).
            "','".$this->escape_string( $authdb->authRemoteAddr()).
            "','".strtoupper( $operation ).
            "','".$this->escape_string( $user_name ).
            "','".$this->escape_string( $group_name ).
            "','".$group_type."')";
        $this->query( $sql );
    }

    /*
     * ============================================
     *   METHODS FOR RETRIEVING THE LOGGED EVENTS
     * ============================================
     */

    /*
     * Return an array of logged records. Each element is an associative
     * array of the following format:
     *
     *   Key            | Type            | Description
     *   ---------------+-----------------+------------------------------------------------------------------------------
     *   id             | unsigned number | the unique identifier of the record in the database
     *   event_time     | LusiTime        | the time of the event
     *   requestor      | string          | user account of a person requested the operation
     *   requestor_host | string          | the host (IP address or DNS name) name from which the operation was requested
     *   operation      | string          | the operation code, allowed values 'ADD', 'DELETE'
     *   user_account   | string          | the user account affcted by the operation
     *   group_name     | string          | the group affected by the operation
     *   group_type     | string          | a type of the group, allowed values 'POSIX', 'NETGROUP' 
     *
     */
    public function get_group_management () {
        $list = array();
        $this->connect();
        $result = $this->query( "SELECT * FROM group_management" );
        $nrows = mysql_numrows( $result );
        for( $i = 0; $i < $nrows; $i++ ) {
            $row = mysql_fetch_array( $result, MYSQL_ASSOC );
            $row['event_time'] = LusiTime::from64($row['event_time']);
            $row['operation']  = strtolower( $row['operation'] );
            $row['class']      = 'group_management';
            array_push ( $list, $row );
        }
        return $list;
    }

    /*
     * ================================
     *   MySQL TRANSACTION MANAGEMENT
     * ================================
     */
    public function begin () {
        $this->connect();
        if( $this->in_transaction ) return;
        $this->transaction( 'BEGIN' );
        $this->in_transaction = true;
    }

    public function commit () {
        $this->connect();
        if( !$this->in_transaction ) return;
        $this->transaction( 'COMMIT' );
        $this->in_transaction = false;
    }

    public function rollback () {
        $this->connect();
        if( !$this->in_transaction ) return;
        $this->transaction( 'ROLLBACK' );
        $this->in_transaction = false;
    }

    private function transaction ( $transition ) {
        if( !mysql_query( $transition, $this->link ))
            throw new AuthDBException (
                __METHOD__,
                "MySQL error: ".mysql_error( $this->link ).', in query: '.$transition );
    }

    /* =================
     *   MySQL QUERIES
     * =================
     */
    public function query ( $sql ) {

        $this->connect();

        if( !$this->in_transaction )
            throw new AuthDBException (
                __METHOD__, "no active transaction" );

        $result = mysql_query( $sql, $this->link );
        if( !$result )
            throw new AuthDBException (
                __METHOD__,
                "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );
        return $result;
    }

    /* ==========================================================
     *   MISC. OPERATIONS REQUIREING DATABASE SERVER CONNECTION
     * ==========================================================
     */
    public function escape_string( $text ) {
        return mysql_real_escape_string( $text, $this->link );  }
 
    /**
     * Make a connection if this hasn't been done yet.
     */
    private function connect () {
        if( !isset( $this->link )) {

            /* Connect to MySQL server and register the shutdown function
             * to clean up after self by doing 'ROLLBACK'. This would allow
             * to use so called 'persistent' MySQL connections.
             * 
             * NOTE: using the 'persistent' connection. This connection won't be
             * closed by 'mysql_close()'.
             */
            $new_link = false; // true;
            $this->link = mysql_pconnect( $this->host, $this->user, $this->password, $new_link );
            if( !$this->link )
                throw new AuthDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).", in function: mysql_connect" );

            if( !mysql_select_db( $this->database, $this->link ))
                throw new AuthDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).", in function: mysql_select_db" );

            $sql = "SET SESSION SQL_MODE='ANSI'";
            if( !mysql_query( $sql, $this->link ))
                throw new AuthDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );

            $sql = "SET SESSION AUTOCOMMIT=0";
            if( !mysql_query( $sql, $this->link ))
                throw new AuthDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );

            register_shutdown_function( array( $this, "rollback" ));
        }
    }
}

/* ==========================
 * UNIT TEST FOR LDAP METHODS
 * ==========================
 *
try {
    $logger = Logger::instance();
    $logger->begin();
    $logger->group_management ( 'add', 'perazzo', 'amo14410' );
    foreach( $logger->get_group_management() as $entry ) {
        print(
            '<br>id             : <b>'.$entry['id'].'</b>'.
            '<br>event_time     : <b>'.$entry['event_time']->toStringShort().'</b>'.
            '<br>requestor      : <b>'.$entry['requestor'].'</b>'.
            '<br>requestor_host : <b>'.$entry['requestor_host'].'</b>'.
            '<br>class          : <b>'.$entry['class'].'</b>'.
            '<br>operation      : <b>'.$entry['operation'].'</b>'.
            '<br>user_account   : <b>'.$entry['user_account'].'</b>'.
            '<br>group_name     : <b>'.$entry['group_name'].'</b>'.
            '<br>group_type     : <b>'.$entry['group_type'].'</b>'
        );
    }
    $logger->commit();

} catch ( Exception $e ) { print e; }
*/

?>
