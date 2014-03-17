/**
 * The application for displaying user-defined run tables
 *
 * @returns {Runtables_User}
 */
function Runtables_User (experiment, access_list) {

    var _that = this ;
    
    // -----------------------------------------
    // Allways call the base class's constructor
    // -----------------------------------------

    FwkApplication.call(this) ;

    // ------------------------------------------------
    // Override event handler defined in the base class
    // ------------------------------------------------

    this.on_activate = function() {
        this.on_update() ;
    } ;

    this.on_deactivate = function() {
        this._init() ;
    } ;

    this.on_update = function () {
        if (this.active) {
            this._init() ;
        }
    } ;

    // -----------------------------
    // Parameters of the application
    // -----------------------------

    this.experiment  = experiment ;
    this.access_list = access_list ;

    // --------------------
    // Own data and methods
    // --------------------

    this._is_initialized = false ;

    this._wa = null ;
    this._tabs = null ;
    this._new_table_creator = null ;
    this._new_table_creator_info = null ;
    this._new_table_creator_columns = null ;

    this._updated = {} ;
    this._tables  = {} ;

    this._table_data = null ;

    this._table_creator_dict = null ;

    this._init = function () {

        if (this._is_initialized) return ;
        this._is_initialized = true ;

        this.container.html('<div id="runtables-user"></div>') ;
        this._wa = this.container.find('div#runtables-user') ;

        if (!this.access_list.runtables.read) {
            this._wa.html(this.access_list.no_page_access_html) ;
            return ;
        }
        
        this._load_tables() ;
    } ;


    /////////////////////////
    /// DISPLAYING TABLES ///
    /////////////////////////


    this._load_tables = function () {

        this._wa.html('Loading...') ;

        Fwk.web_service_GET (
            '../portal/ws/runtable_user_tables.php' ,
            {exper_id: this.experiment.id} ,
            function (data) {
                _that._table_data = data.table_data ;
                _that._display_tables() ;
            } ,
            function (msg) {
                _that._wa.html(msg) ;
            }
        ) ;
    } ;

    this._display_tables = function () {

        var html =
'<div id="tabs">' +
'  <ul>' ;
        for (var i in this._table_data) {
            var table_data = this._table_data[i] ;
            var table_id   = table_data.config.id ;
            var table_name = table_data.config.name ;
            html +=
'    <li><a href="#'+table_id+'">'+table_name+'</a></li>' ;
        }
        html +=
'    <li><a href="#add" title="Add new table" ><span class="ui-icon ui-icon-plus"></span></a></li>' +
'  </ul>' ;
        for (var i in this._table_data) {
            var table_data = this._table_data[i] ;
            var table_id   = table_data.config.id ;
            html +=
'  <div id="'+table_id+'">' +
'    <div id="ctrl">' +
'      <div class="group">' +
'        <span class="label">Runs</span>' +
'        <select class="update-trigger" name="'+table_id+':runs" >' +
'          <option value="20" >last 20</option>' +
'          <option value="100">last 100</option>' +
'          <option value="200">last 200</option>' +
'          <option value="range">specific range</option>' +
'          <option value="all">all</option>' +
'        </select>' +
'      </div>' +
'      <div class="group">' +
'        <span class="label">First</span>' +
'        <input class="update-trigger" type="text" name="'+table_id+':first" value="-20" size=8 disabled />' +
'        <span class="label">Last</span>' +
'        <input class="update-trigger" type="text" name="'+table_id+':last"  value="" size=8 disabled />' +
'      </div>' +
'      <div class="buttons">' +
'        <button class="control-button" name="'+table_id+':search" title="search and display results" >Search</button>' +
'        <button class="control-button" name="'+table_id+':reset"  title="reset the form"             >Reset</button>' +
'      </div>' +
'      <div class="buttons">' +
'        <button class="control-button edit"   name="'+table_id+':edit"    title="edit table configuration: add/remove/remove/rearrange columns, rename the table, etc." >Reconfigure</button>' +
'        <button class="control-button delete" name="'+table_id+':delete"  title="delete the table" >Delete</button>' +
'      </div>' +
'      <div style="clear:both;"></div>' +
'    </div>' +
'    <div id="body">' +
'      <div class="info" id="info"    style="float:left;">&nbsp;</div>' +
'      <div class="info" id="updated" style="float:right;">&nbsp;</div>' +
'      <div style="clear:both;"></div>' +
'      <div id="table" class="table" ></div>' +
'    </div>' +
'  </div>' ;
        }
        html +=
'  <div id="add"></div>' +
'</div>' ;

        this._wa.html(html) ;
        this._tabs = this._wa.children('#tabs').tabs() ;

        this._tabs.find('.control-button').button().click(function () {
            var s2 = this.name.split(':') ;
            var table_id = s2[0] ;
            var op = s2[1] ;
            switch (op) {
                case 'search' : _that._load  (table_id) ; break ;
                case 'reset'  : _that._reset (table_id) ; break ;
                case 'edit'   : _that._edit  (table_id) ; break ;
                case 'delete' : _that._delete(table_id) ; break ;
            }
        }) ;
        this._tabs.find('select.update-trigger').change(function () {
            var s2 = this.name.split(':') ;
            var table_id = s2[0] ;
            var range    = $(this).val() ;
            if (range === 'range') {
                _that._tabs.find('input.update-trigger[name="'+table_id+':first"]').removeAttr('disabled') ;
                _that._tabs.find('input.update-trigger[name="'+table_id+':last"]') .removeAttr('disabled') ;
            } else {
                _that._tabs.find('input.update-trigger[name="'+table_id+':first"]').attr('disabled', 'disabled') ;
                _that._tabs.find('input.update-trigger[name="'+table_id+':last"]') .attr('disabled', 'disabled') ;
                switch (range) {
                    case '20' :
                    case '100' :
                    case '200' :
                        _that._tabs.find('input[name="'+table_id+':first"]').val(-parseInt(range)) ;
                        _that._tabs.find('input[name="'+table_id+':last"]').val('') ;
                        break ;
                    case 'all' :
                        _that._tabs.find('input[name="'+table_id+':first"]').val('') ;
                        _that._tabs.find('input[name="'+table_id+':last"]').val('') ;
                        break ;
                }
            }
        }) ;
        this._tabs.find('.update-trigger').change(function () {
            var s2 = this.name.split(':') ;
            var table_id = s2[0] ;
            _that._load(table_id) ;
        }) ;
        for (var i in this._table_data) {
            var table_name = this._table_data[i] ;
            var table_id = table_name.config.id ;
            this._updated[table_id] = this._tabs.find('div#'+table_id).find('#updated') ;
            this._tables [table_id] = this._init_table(table_id) ;
            this._load(table_id) ;
        }
        this._init_table_creator('add') ;
    } ;

    this._init_table = function (table_id)  {

        var table_data = this._table_data[table_id] ;

        var tab_body   = this._tabs.find('div#'+table_id) ;
        var table_cont = tab_body.find('div#table') ;

        var hdr = [
            'RUN'
        ] ;
        for (var i in table_data.config.coldef) {
            var col = table_data.config.coldef[i] ;
            hdr.push(col.name) ;
        }
        var rows = [] ;
        var num_hdr_rows = 2 ;
        var max_hdr_rows = 5 ;
        var table = new SmartTable (
            table_cont ,
            hdr ,
            rows ,
            num_hdr_rows ,
            max_hdr_rows
        ) ;
        this._tables[table_id] = table ;

        return table ;
    } ;

    this._display = function (table_id) {

        var title = 'show the run in the e-Log Search panel within the current Portal' ;

        var table      = this._tables    [table_id] ;
        var table_data = this._table_data[table_id] ;

        var rows = [] ;
        for (var run in table_data.runs) {
            var row = [] ;
            row.push(
                '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="lb_link">'+run+'</a>'
            ) ;
            var param2value = table_data.runs[run] ;
            for (var i in table_data.config.coldef) {
                var col = table_data.config.coldef[i] ;
                var name  = col.name ;
                var value = name in param2value ? param2value[name] : '' ;

                // TODO: replace with lazy initialization using JQuery to
                //       prevent markup screwup due to special symbols.

                if (col.is_editable) {
                    var run_id = table_data.run2id[run] ;
                    row.push('<input class="editable" coldef_id="'+col.id+'" run_id="'+run_id+'" type="text" value="'+value+'" />') ;
                } else {
                    row.push(value) ;
                }
            }
            rows.push(row) ;
        }
        rows.reverse() ;
        table.load(rows) ;

        // Set up event handlers for updating cell

        var tab_body   = this._tabs.find('div#'+table_id) ;
        var table_cont = tab_body.find('div#table') ;

        table_cont.find('input.editable').change(function() {

            var elem      = $(this) ;
            elem.addClass('modified') ;

            var coldef_id = parseInt(elem.attr('coldef_id')) ;
            var run_id    = parseInt(elem.attr('run_id')) ;
            var value     = elem.val() ;

            _that._updated[table_id].html('Saving...') ;

            _that._update_cell (
                table_id, run_id, coldef_id, value,
                function (data) {
                    elem.removeClass('modified') ;
                    _that._updated[table_id].html('[ Last update on: <b>'+data.updated+'</b> ]') ;
                }
            ) ;
        }) ;
    } ;

    this._load = function (table_id) {

        this._updated[table_id].html('Loading...') ;

        var params = {
            exper_id: this.experiment.id ,
            table_id: table_id
        } ;
        var range = this._tabs.find('select[name="'+table_id+':runs"]').val() ;
        switch (range) {
            case '20' :
            case '100' :
            case '200' :
                params.from_run    = -parseInt(range) ;
                params.through_run = 0 ;
                break ;
            case 'range' :
                params.from_run    = parseInt(this._tabs.find('input[name="'+table_id+':first"]').val()) ;
                params.through_run = parseInt(this._tabs.find('input[name="'+table_id+':last"]').val()) ;
                break ;
            case 'all' :
            default :
                params.from_run    = 0 ;
                params.through_run = 0 ;
                break ;
        }

        Fwk.web_service_GET (
            '../portal/ws/runtable_user_table_get.php' ,
            params ,
            function (data) {
                _that._table_data[table_id].runs   = data.runs ;
                _that._table_data[table_id].run2id = data.run2id ;
                _that._display(table_id) ;
                _that._updated[table_id].html('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }
        ) ;
    } ;

    this._update_cell = function (table_id, run_id, coldef_id, value, on_success) {
        var cells = [{
            run_id:    run_id ,
            coldef_id: coldef_id ,
            value:     value
        }] ;
        var params = {
            exper_id: this.experiment.id ,
            table_id: table_id ,
            cells:    JSON.stringify(cells)
        } ;
        Fwk.web_service_POST (
            '../portal/ws/runtable_user_update.php' ,
            params ,
            on_success
        ) ;
    } ;


    ///////////////////////////////////////
    /// DIALOGS FOR CREATING NEW TABLES ///
    ///////////////////////////////////////


    this._init_table_creator = function (table_id) {

        this._new_table_creator = this._tabs.find('div#'+table_id) ;

        var html =
'<div id="ctrl">' +
'  <div class="group">' +
'    <p><b>Instructions:</b> Fill in a desired table name and configure the initial columns before hitting' +
'       the <b>Save</b> button. Note that more columns can be added later using the table manager' +
'       dialog when viewing the table. If no columns are provided when creating the table then the table' +
'       will initially have just one - for run numbers. Columns can also be removed later.' +
'    </p>' +
'    <div>' +
'      <span class="label">Table name</span>' +
'      <input type="text" size="50" name="name" />' +
'      <span class="label">is private</span>' +
'      <input type="checkbox" name="private" />' +
'      <span class="comment">private tables can only be modified by their owners - users who created the tables</span>' +
'    </div>' +
'  </div>' +
'  <div class="buttons">' +
'    <button name="create" title="create the table" >Save</button>' +
'    <button name="reset"  title="reset the table creation form" >Reset</button>' +
'  </div>' +
'  <div style="clear:both;"></div>' +
'</div>' +
'<div id="'+table_id+'-tabs">' +
'  <ul>' +
'    <li><a href="#config">Configuration</a></li>' +
'    <li><a href="#preview">Preview</a></li>' +
'  </ul>' +
'  <div id="config">' +
'    <div id="initial-columns">' +
'      <div class="info" id="info" style="float:right;">&nbsp;</div>' +
'      <div style="clear:both;"></div>' +
'      <table class="columns" border="0" cellspacing="0">' +
'        <thead>' +
'          <tr>' +
'            <td>&nbsp;</td>' +
'            <td>Column Name</td>' +
'            <td>Type</td>' +
'            <td>Data Source</td>' +
'          </tr>' +
'        </thead>' +
'        <tbody></tbody>' +
'      </table>' +
'    </div>' +
'  </div>' +
'  <div id="preview">' +
'    <div id="table" class="table" ></div>' +
'  </div>' +
'</div>'  ;
        this._new_table_creator.html(html) ;

        this._new_table_creator.find('#ctrl').find('button').button().click(function () {
            switch (this.name) {
                case 'create' : _that._create_new_table() ; break ;
                case 'reset'  : _that._reset_table_creator() ; break ;
            }
        }) ;
        this._new_table_creator.find('#'+table_id+'-tabs').tabs() ;

        this._new_table_creator_info    = this._new_table_creator.find('div#info') ;
        this._new_table_creator_columns = this._new_table_creator.find('table.columns') ;

        this._load_table_creator_dictionary() ;
    } ;

    /**
     * Load the dictionary of column types/sources and then qadd teh very first
     * placeholder to the table.
     * 
     * @returns {undefined}
     */
    this._load_table_creator_dictionary = function () {
        Fwk.web_service_GET (
            '../portal/ws/runtable_user_dict_get.php' ,
            {exper_id: this.experiment.id} ,
            function (data) {
                _that._table_creator_dict = data.types ;
                _that._add_new_column_paceholder_if_needed() ;
            }
        ) ;
    } ;

    this._create_new_table = function () {

        var name = this._new_table_creator.find('input[name="name"]').val() ;
        if (name === '') {
            Fwk.report_error('Please, fill in the table name field and try again!') ;
            return ;
        }

        var coldef = [] ;
        this._new_table_creator_columns.find('tr.coldef').each(function () {
            var tr = $(this) ;
            if (tr.hasClass('spare')) return ;
            coldef.push ({
                name:     tr.find('input[name="column"]').val() ,
                type:     tr.find('select[name="type"]').val() ,
                source:   tr.find('select[name="source"]').val() ,
                position: tr.attr('pos')
            }) ;
        }) ;
        if (!coldef.length) {
            Fwk.report_error('Please, provide at least one column definition and try again!') ;
            return ;
        }

        var button_create = this._new_table_creator.find('button[name="create"]').button() ;
        button_create.button('disable') ;

        var params = {
            exper_id: this.experiment.id ,
            name:       name ,
            is_private: this._new_table_creator.find('input[name="private"]').attr('checked') ? 1 : 0 ,
            coldef:     JSON.stringify(coldef)
        } ;
        Fwk.web_service_POST (
            '../portal/ws/runtable_user_create.php' ,
            params ,
            function () {
                _that._load_tables() ;    // reload all tables
                                            // this sshould also reset the table creator
            } ,
            function (msg) {
                Fwk.report_error(msg) ;
                button_create.button('enable') ;
            }
        ) ;
    } ;

    this._reset_table_creator = function () {
        this._new_table_creator.find('input[name="name"]').val('') ;
        this._new_table_creator_columns.find('tr.coldef').each(function () {
            var tr = $(this) ;
            var row = parseInt(tr.attr('row')) ;
            if (row) tr.remove() ;
        }) ;
        this._clear_table_creator_column(0) ;
    } ;
 
    this._clear_table_creator_column = function (row) {
        var tr = this._new_table_creator_columns.find('tr.coldef[row="'+row+'"]') ;
        if (tr.length) {
            tr.find('input[name="column"]').val('') ;
            tr.find('select[name="type"]').val('') ;
            tr.find('select[name="source"]').html('').attr('disabled', 'disabled') ;
            tr.addClass('spare') ;
            this._update_table_creator_info() ;

        }
        this._add_new_column_paceholder_if_needed() ;
    } ;

    this._left_table_creator_column = function (row) {
        var tr = this._new_table_creator_columns.find('tr.coldef[row="'+row+'"]') ;
        if (tr.length) {
            console.log('_left_table_creator_column: tr.length='+tr.length) ;
            var tr_prev = tr.prev() ;
            if (tr_prev.length) {

                var pos      = tr     .attr('pos') ;
                var pos_prev = tr_prev.attr('pos') ;

                console.log('_left_table_creator_column: pos='+pos+' pos_prev='+pos_prev) ;

                tr     .attr('pos', pos_prev) ;
                tr_prev.attr('pos', pos) ;

                tr.after(tr_prev) ;
                
                this._update_new_table_creator_events() ;
                this._update_table_creator_info() ;
            }
        }
    } ;

    this._right_table_creator_column = function (row) {
        var tr = this._new_table_creator_columns.find('tr.coldef[row="'+row+'"]') ;
        if (tr.length) {
            console.log('_right_table_creator_column: tr.length='+tr.length) ;
            var tr_next = tr.next() ;
            if (tr_next.length) {

                var pos      = tr     .attr('pos') ;
                var pos_next = tr_next.attr('pos') ;

                console.log('_right_table_creator_column: pos='+pos+' pos_next='+pos_next) ;

                tr     .attr('pos', pos_next) ;
                tr_next.attr('pos', pos) ;

                tr_next.after(tr) ;
                
                this._update_new_table_creator_events() ;
                this._update_table_creator_info() ;
            }
        }
    } ;

    this._new_column_paceholder = function (row, position) {
        var html =
'<tr class="coldef spare" row="'+row+'" pos="'+position+'">' +
'  <td>' +
'      <button name="clear" style="font-size:8px;" title="remove this column from table definition" ><span class="ui-icon ui-icon-closethick"></span></button>' +
'      <button name="left"  style="font-size:8px;" title="move this column up"                      ><span class="ui-icon ui-icon-arrowthick-1-n"></span></button>' +
'      <button name="right" style="font-size:8px;" title="move this column down"                    ><span class="ui-icon ui-icon-arrowthick-1-s"></span></button>' +
'  </td>' +
'  <td name="column" ><input  name="column" type="text" size="40" title="the name of the column as it will appear in the table" val="" /></td>' +
'  <td name="type"   ><select name="type"                         title="the type of the column" /></select></td>' +
'  <td name="source" ><select name="source"                       title="the data source for the cells" /></select></td>' +
'</tr>' ;
        return html ;
    } ;

    /**
     * Add more column definitions if we're run out of the last spare
     * 
     * @returns {undefined}
     */
    this._add_new_column_paceholder_if_needed = function () {

        var tr_spare = this._new_table_creator_columns.find('tr.spare') ;
        if (tr_spare.length) return ;

        var tr_last = this._new_table_creator_columns.find('tr.coldef:last') ;
        var row = undefined ;
        if (tr_last.length) {
            row = parseInt(tr_last.attr('row')) + 1 ;
            var position = this._new_table_creator_columns.find('tr.coldef').length ;
            tr_last.after(this._new_column_paceholder(row, position)) ;
        } else {
            row = 0 ;
            var position = 0 ;
            this._new_table_creator_columns.find('tbody').html(this._new_column_paceholder(row, position)) ;
        }
        this._update_new_table_creator_events() ;   // make sure all event handlers are activated
        this._apply_table_creator_dict(row) ;       // initialize selectors in the row
        this._update_table_creator_info() ;
    } ;

    this._update_new_table_creator_events = function () {

        // ATTENTION: note all previously set event handlers are removed to avoid
        //            multiple execution!

        this._new_table_creator_columns.find('button[name="clear"]').button().off().click(function () {
            var row =  parseInt($(this).closest('tr').attr('row')) ;
            _that._clear_table_creator_column(row) ;
        }) ;
        this._new_table_creator_columns.find('button[name="left"]').button().off().click(function () {
            var row =  parseInt($(this).closest('tr').attr('row')) ;
            _that._left_table_creator_column(row) ;
        }) ;
        this._new_table_creator_columns.find('button[name="right"]').button().off().click(function () {
            var row =  parseInt($(this).closest('tr').attr('row')) ;
            _that._right_table_creator_column(row) ;
        }) ;
        this._new_table_creator_columns.find('input[name="column"]').off().change(function () {
            var elem = $(this) ;
            var tr = elem.closest('tr') ;
            var column = elem.val() ;
            if (column === '')
                tr.addClass('spare') ;
            else {
                tr.removeClass('spare') ;
                _that._add_new_column_paceholder_if_needed() ;
            }
            _that._update_table_creator_info() ;
        }) ;
        this._new_table_creator_columns.find('select[name="type"]').off().change(function () {
            var elem = $(this) ;
            var tr = elem.closest('tr') ;
            var row =  parseInt(tr.attr('row')) ;
            var select_type = elem.val() ;
            if (select_type === '') _that._apply_table_creator_dict(row) ;
            else                    _that._apply_table_creator_dict(row, select_type) ;
            _that._add_new_column_paceholder_if_needed() ;
        }) ;
        this._new_table_creator_columns.find('select[name="source"]').off().change(function () {
            var elem = $(this) ;
            var tr = elem.closest('tr') ;
            var select_source = elem.val() ;
            tr.find('input[name="column"]').val(select_source) ;
            if (select_source === '') {
                tr.addClass('spare') ;
            } else {
                tr.removeClass('spare') ;
                _that._add_new_column_paceholder_if_needed() ;
            }
            _that._update_table_creator_info() ;
        }) ;
    } ;
    this._update_table_creator_info = function () {
        var tr = this._new_table_creator_columns.find('tr.coldef') ;
        var named = 0 ;
        var editable = 0 ;
        tr.each(function () {
            var elem = $(this) ;
            if (!elem.hasClass('spare')) named++ ;
            if (elem.find('select[name="type"]').val() === 'Editable') editable++ ;
        }) ;
        var html = '<b>'+named+'</b> columns [ EDITABLE: <b>'+editable+'</b> ]' ;
        this._new_table_creator_info.html(html) ;

        this._update_table_preview() ;
    } ;

    /**
     * Apply the dictonary in the specified context.
     * 
     * The context:
     * - all columns if both 'row' and 'select_type' are 'undefined'
     * - the specified row if 'row' is not 'undefined'
     * - both type and source of the specified row if both 'row' and 'select_type' are not 'undefined'
     *
     * @param {type} row
     * @param {type} select_type
     * @returns {undefined}
     */
    this._apply_table_creator_dict = function (row, select_type) {
        var tr = this._new_table_creator_columns.find(row === undefined ? 'tr.coldef' : 'tr.coldef[row="'+row+'"]') ;
        if (tr.length) {
            var html = '<option value="">&lt;select&gt;</option>' ;
            for (var type in this._table_creator_dict) {
                html += '<option>'+type+'</option>' ;
            }
            var type_elem = tr.find('select[name="type"]') ;
            type_elem.html(html) ;
            var source_elem = tr.find('select[name="source"]') ;
            if (select_type !== undefined) {
                type_elem.val(select_type) ;
                if (this._table_creator_dict[select_type].length) {
                    var html = '<option value="">&lt;select&gt;</option>' ;
                    for (var i in this._table_creator_dict[select_type]) {
                        var source = this._table_creator_dict[select_type][i] ;
                        html += '<option>'+source+'</option>' ;
                    }
                    source_elem.html(html) ;
                    source_elem.removeAttr('disabled') ;
                } else {
                    source_elem.html('') ;
                    source_elem.attr('disabled', 'disabled') ;
                }
            } else {
                source_elem.html('') ;
                source_elem.attr('disabled', 'disabled') ;
            }
        }
    } ;


    this._reset = function (table_id) {
        var tab_body = this._tabs.find('div#'+table_id) ;
        tab_body.find('.update-trigger').val('') ;
        this._load(table_id) ;
    } ;
    this._delete = function (table_id) {

        var tab_body   = this._tabs.find('div#'+table_id) ;
        var button_delete = tab_body.find('button[name="delete"]').button() ;
        button_delete.button('disable') ;

        Fwk.ask_yes_no (
            'Confirm Destructive Operation' ,
            'Would you really like to delete the table? '+
            'Note that after pressing <b>Yes</b> the information stored in the user-editable columns will be destroyed in the database.' ,
            function () {
                Fwk.web_service_GET (
                    '../portal/ws/runtable_user_delete.php' ,
                    {exper_id: _that.experiment.id, table_id: table_id} ,
                    function () {
                        _that._load_tables() ;
                    } ,
                    function (msg) {
                        Fwk.report_error(msg) ;
                        button_delete.button('enable') ;
                    }
                ) ;
            }
        ) ;
    } ;
    this._edit = function (table_id) {
        console.log('_delete: not implemented, make sure the table can be edited by the current user/owner of teh table') ;
    } ;
    this._update_table_preview = function ()  {

        var table_cont = this._new_table_creator.find('div#table.table') ;

        var hdr = [
            'RUN'
        ] ;
        var rows = [] ;

        var trs = this._new_table_creator_columns.find('tr.coldef') ;
        if (trs.length) {
            trs.each(function () {
                var tr = $(this) ;
                if (!tr.hasClass('spare')) {
                    var name  = tr.find('input[name="column"]').val() ;
                    hdr.push(name) ;
                }
            }) ;
            for (var i=123; i<133; i++) {
                var row = [
                    i
                ] ;
                var j = 0 ;
                trs.each(function () {
                    var tr = $(this) ;
                    if (!tr.hasClass('spare')) {
                        var type  = tr.find('select[name="type"]').val() ;
                        switch (type) {
                            case 'Editable' :
                                row.push('<input type="text" value="edit me" >') ;
                                break ;
                            default :
                                row.push(++j) ;
                                break ;
                        }
                    }
                }) ;
                rows.push(row) ;
            }
        }
        var num_hdr_rows = 2 ;
        var max_hdr_rows = 5 ;
        var table = new SmartTable (
            table_cont ,
            hdr ,
            rows ,
            num_hdr_rows ,
            max_hdr_rows
        ) ;
        table.display() ;
    } ;
}
define_class (Runtables_User, FwkApplication, {}, {});
