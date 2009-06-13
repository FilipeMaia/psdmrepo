<?php

/**
 * Class RegDBConnection encapsulates operations with the database
 */
class RegDBConnection {

    /* Data members
     */
    private $link;
    private $host;
    private $user;
    private $password;
    private $database;
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
        $this->host     = $host;
        $this->user     = $user;
        $this->password = $password;
        $this->database = $database;
    }

    /**
     * Destructor
     */
    public function __destruct () {
        if( isset( $this->link ))
            mysql_close( $this->link );
    }

    /*
     * ==========================
     *   TRANSACTION MANAGEMENT
     * ==========================
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
        if( !mysql_query( $transition ))
            throw new RegDBException (
                __METHOD__,
                "MySQL error: ".mysql_error().', in query: '.$transition );
    }

    /* ===========
     *   QUERIES
     * ===========
     */
    public function query ( $sql ) {

        $this->connect();

        if( !$this->in_transaction )
            throw new RegDBException (
                __METHOD__, "no active transaction" );

        $result = mysql_query( $sql );
        if( !$result )
            throw new RegDBException (
                __METHOD__,
                "MySQL error: ".mysql_error().', in query: '.$sql );
        return $result;
    }

    /**
     * Make a connection if this hasn't been done yet.
     */
    private function connect () {
        if( !isset( $this->link )) {
            $this->link = mysql_connect( $this->host, $this->user, $this->password );
            if( !$this->link )
                throw new RegDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error().", in function: mysql_connect" );

            if( !mysql_select_db( $this->database ))
                throw new RegDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error().", in function: mysql_select_db" );

            $sql = "SET SESSION SQL_MODE='ANSI'";
            if( !mysql_query( $sql ))
                throw new RegDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error().', in query: '.$sql );

            $sql = "SET SESSION AUTOCOMMIT=0";
            if( !mysql_query( $sql ))
                throw new RegDBException (
                    __METHOD__,
                    "MySQL error: ".mysql_error().', in query: '.$sql );
        }
    }
}
?>
