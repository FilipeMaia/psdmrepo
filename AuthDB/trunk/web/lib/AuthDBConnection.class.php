<?php

namespace AuthDB;

require_once( 'AuthDB.inc.php' );

/**
 * Class AuthDBConnection encapsulates operations with the database
 */
class AuthDBConnection {

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

$conn = new AuthDBConnection (
    AUTHDB_DEFAULT_HOST,
    AUTHDB_DEFAULT_USER,
    AUTHDB_DEFAULT_PASSWORD,
    AUTHDB_DEFAULT_DATABASE );

try {
    $conn->begin();
    $conn->rollback();

 } catch ( AuthDBException $e ) {
    print( e.toHtml());
}
*/
?>
