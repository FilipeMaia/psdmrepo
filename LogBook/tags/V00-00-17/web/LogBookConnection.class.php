<?php

/**
 * Class LogBookConnection encapsulates operations with the database
 */
class LogBookConnection {

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
        if( !mysql_query( $transition, $this->link ))
            throw new LogBookException(
                __METHOD__,
                "MySQL error: ".mysql_error( $this->link ).', in query: '.$transition );
    }

    /* ===========
     *   QUERIES
     * ===========
     */
    public function query ( $sql ) {

        $this->connect();

        if( !$this->in_transaction )
            throw new LogBookException(
                __METHOD__, "no active transaction" );

        $result = mysql_query( $sql, $this->link );
        if( !$result )
            throw new LogBookException(
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
            $new_link = true;
            $this->link = mysql_connect( $this->host, $this->user, $this->password, $new_link );
            if( !$this->link )
                throw new LogBookException(
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).", in function: mysql_connect" );

            if( !mysql_select_db( $this->database, $this->link ))
                throw new LogBookException(
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).", in function: mysql_select_db" );

            $sql = "SET SESSION SQL_MODE='ANSI'";
            if( !mysql_query( $sql, $this->link ))
                throw new LogBookException(
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );

            $sql = "SET SESSION AUTOCOMMIT=0";
            if( !mysql_query( $sql, $this->link ))
                throw new LogBookException(
                    __METHOD__,
                    "MySQL error: ".mysql_error( $this->link ).', in query: '.$sql );
        }
    }
}
?>
