<!--
The page for creating displaying all experiments.
-->
<!DOCTYPE HTML PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN">
<html>
    <head>
        <meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
        <title>Display Experiments</title>
    </head>
    <style type="text/css">
      body {
        margin:0;
        padding:0; }
    </style>

    <link rel="stylesheet" type="text/css" href="/yui/build/fonts/fonts-min.css" />
    <link rel="stylesheet" type="text/css" href="/yui/build/paginator/assets/skins/sam/paginator.css" />
    <link rel="stylesheet" type="text/css" href="/yui/build/datatable/assets/skins/sam/datatable.css" />
    <link rel="stylesheet" type="text/css" href="/yui/build/button/assets/skins/sam/button.css" />
    <link rel="stylesheet" type="text/css" href="/yui/build/tabview/assets/skins/sam/tabview.css" />

    <script type="text/javascript" src="/yui/build/yahoo-dom-event/yahoo-dom-event.js"></script>
    <script type="text/javascript" src="/yui/build/connection/connection-min.js"></script>
    <script type="text/javascript" src="/yui/build/json/json-min.js"></script>
    <script type="text/javascript" src="/yui/build/element/element-min.js"></script>
    <script type="text/javascript" src="/yui/build/paginator/paginator-min.js"></script>
    <script type="text/javascript" src="/yui/build/datasource/datasource-min.js"></script>
    <script type="text/javascript" src="/yui/build/datatable/datatable-min.js"></script>
    <script type="text/javascript" src="/yui/build/button/button-min.js"></script>
    <script type="text/javascript" src="/yui/build/tabview/tabview-min.js"></script>

    <script type="text/javascript" src="/yui/build/yahoo/yahoo-min.js"></script>
    <script type="text/javascript" src="/yui/build/dom/dom-min.js"></script>

    <!--begin custom header content for this example-->
    <style type="text/css">

      .yui-skin-sam .yui-dt-liner {
        white-space:nowrap; }

      #main {
        margin-left:2em;
        margin-right:2em; }

      #tabs, #pushbutton1  {
        margin-left:2em;
        margin-right:2em; }

      #group_table          table,
      #instrument_table     table,
      #experiment_table     table,
      #run_table            table,
      #run_param_table      table,
      #run_val_table        table,
      #run_val_int_table    table,
      #run_val_double_table table,
      #run_val_text_table   table {

        margin-left:auto;
        margin-right:auto; }

      #group_table,          #group_table          .yui-dt-loading,
      #instrument_table,     #instrument_table     .yui-dt-loading,
      #experiment_table,     #experiment_table     .yui-dt-loading,
      #run_table,            #run_table            .yui-dt-loading,
      #run_param_table,      #run_param_table      .yui-dt-loading,
      #run_val_table,        #run_val_table        .yui-dt-loading,
      #run_val_int_table,    #run_val_int_table    .yui-dt-loading,
      #run_val_double_table, #run_val_double_table .yui-dt-loading,
      #run_val_text_table,   #run_val_text_table   .yui-dt-loading {

        text-align:center;
        background-color:transparent; }

      /* begin tab navigation */

      #tabnav {
        position:relative;
        /*width:800px;*/
        margin:10px 0px 0px 0px;
        padding:0 0 8px 0;
        line-height:normal;
        border-bottom:2px solid #0071bc;
        height:24px }

      #tabnav ul {
        margin:0px;
        padding:0px;
        list-style:none; }

      #tabnav li {
        float:left;
        margin:0px;
        padding:0px; }

      #tabnav a {
        display:block;
        color:#000;
        padding:5px 0 0 0;
        text-decoration:none;
        font-size:13px;
        text-align:center;
        margin-left:70px; }

      #tabnav li a#first {
        margin-left:0px; }

      #tabnav a:hover {
        color:#0071bc; }

      /* end tab navigation */

    </style>
    <!--end custom header content for this example-->

    <link rel="stylesheet" type="text/css" href="RegDBTest.css" />

    <body class="yui-skin-sam">
        <p id="title"><b>Experiments</b></p>
        <div class="yui-content">
          <div id="experiment">
            <p>The table stores the information about experiments.</p>
            <div id="experiment_table"></div>
            <p>Here goes the footer of the tab. We may put additional instructions
               or command buttons in here.</p>
          </div>

    <!-- DYNAMIC CONTENTS GENERATION -->

    <script type="text/javascript"> 

    /* The customizable object for encapsulating the context and
     * operations with tables.
     */
    function Table( itsTableName, itsColumnDefs ) {
        this.name = itsTableName;
        this.columnDefs = itsColumnDefs;
        this.fieldsDefs = [];
        for(i=0; i < itsColumnDefs.length; i++)
            this.fieldsDefs.push( itsColumnDefs[i].key );
        this.dataSource = new YAHOO.util.DataSource(
            "assets/php/scimd_query.php?table="+itsTableName );
        this.dataSource.responseType = YAHOO.util.DataSource.TYPE_JSON;
        this.dataSource.connXhrMode = "queueRequests";
        this.dataSource.responseSchema = {
            resultsList: "ResultSet.Result",
            fields: this.fieldsDefs };
        this.dataTable = new YAHOO.widget.DataTable(
            this.name+"_table",
            this.columnDefs,
            this.dataSource,
            { paginator: new YAHOO.widget.Paginator( { rowsPerPage: 10 } ),
              initialRequest: "" } );

        this.refreshTable = function() {
            this.dataSource.sendRequest(
                "",
                { success: function() {
                      this.set( "sortedBy", null);
                      this.onDataReturnReplaceRows.apply( this, arguments );
                  },
                  failure: function() {
                      this.showTableMessage(
                          YAHOO.widget.DataTable.MSG_ERROR,
                          YAHOO.widget.DataTable.CLASS_ERROR );
                      this.onDataReturnAppendRows.apply( this, arguments );
                  },
                  scope: this.dataTable } ); };
    }

    /* The following code will be executed only when the page is loaded
     * and DOM is initialized.
     */
    YAHOO.util.Event.onDOMReady( function() {

        /* Create and configure the tables.
         */
        var tables = {

            experiment: new Table( "experiment",
                [ { key: "id",         sortable: true, resizeable: true, formatter: YAHOO.widget.DataTable.formatNumber },
                  { key: "name",       sortable: true, resizeable: true },
                  { key: "instr_id",   sortable: true, resizeable: true, formatter: YAHOO.widget.DataTable.formatNumber },
                  { key: "group_id",   sortable: true, resizeable: true, formatter: YAHOO.widget.DataTable.formatNumber },
                  { key: "begin_time", sortable: true, resizeable: true, formatter: YAHOO.widget.DataTable.formatDate,
                                                                         sortOptions: { defaultDir: YAHOO.widget.DataTable.CLASS_DESC } },
                  { key: "end_time",   sortable: true, resizeable: true, formatter: YAHOO.widget.DataTable.formatDate,
                                                                         sortOptions: { defaultDir: YAHOO.widget.DataTable.CLASS_DESC } } ] )
        };

        /* The function to refresh all registered tables.
         */
        function refreshAllTables() {
            tables.experiment.refreshTable();
        }
        refreshAllTables();
    });

    </script>




















            <table border="0" cellpadding="4px" cellspacing="0px" width="100%" align="center">
                <thead style="color:#0071bc;">
                    <th align="left" style="width:96px;">
                        <b>Instrument</b></th>
                    <th align="left">
                        <b>Experiment</b></th>
                    <th align="left">
                        <b>Begin Time</b></th>
                    <th align="left">
                        <b>End Time</b></th>
                    <th align="left">
                        <b>Description</b></th>
                </thead>
                <tbody>
                    <tr>
                        <td><hr></td>
                        <td><hr></td>
                        <td><hr></td>
                        <td><hr></td>
                        <td><hr></td>
                    </tr>
                    <?php
                    require_once('RegDB.inc.php');
                    try {
                        $regdb = new RegDB();
                        $regdb->begin();
                        $instruments = $regdb->instruments();
                        foreach( $instruments as $i ) {
                            $experiments = $regdb->experiments_for_instrument( $i->name());
                            foreach( $experiments as $e ) {
                                //$description = substr( $e->description(), 0, 128 );
                                $description = $e->description();
                                $begin_time = $e->begin_time()->toStringShort();
                                $end_time = $e->end_time()->toStringShort();
                                echo <<< HERE
                    <tr>
                        <td align="left" valign="top">
                            <a href="DisplayInstrument.php?id={$i->id()}"><b>{$i->name()}</b></a></td>
                        <td align="left" valign="top" style="width:10em;">
                            <a href="DisplayExperiment.php?id={$e->id()}"><b>{$e->name()}</b></a></td>
                        <td  valign="top" style="width:10em;">
                            {$begin_time}</td>
                        <td  valign="top" style="width:10em;">
                            {$end_time}</td>
                        <td  valign="top">
                            <i>{$description}</i></td>
                    </tr>
                    <tr>
                        <td></td>
                        <td></td>
                        <td></td>
                        <td></td>
                        <td></td>
                    </tr>
HERE;
                            }
                        }
                        $regdb->commit();

                    } catch ( RegDBException $e ) {
                        print( $e->toHtml());
                    }
                    ?>
                </tbody>
            </table>
        </div>
    </body>
</html>

