define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/SmartTable', 'webfwk/FwkApplication', 'webfwk/Fwk'] ,

function (
    cssloader ,
    Class, SmartTable, FwkApplication, Fwk) {

    cssloader.load('../portal/css/Runtables_User.css') ;

    /**
     * The base class for table dialogs
     *
     * @param {Runtables_User} parent
     * @param {object} table_data
     * @returns {Runtable_Dialog}
     */
    function Runtable_Dialog (parent, table_data) {

        var _that = this ;

        // ----------
        // Parameters
        // ----------

        this._parent = parent ;
        this._table_data = table_data ;

        // --------------------
        // Own data and methods
        // --------------------

        // -- The tab's header (if it needs to be modified) --

        this._cont_hdr = null ;
        this._get_cont_hdr = function () {
            if (!this._cont_hdr) {
                this._cont_hdr = this._parent._tabs.find('ul li a[href="#'+this._table_data.config.id+'"]') ;
            }
            return this._cont_hdr ;
        } ;

        // -- the container for the table and its controls --

        this._cont = null ;
        this._get_cont = function () {
            if (!this._cont) {
                this._cont = this._parent._tabs.find('div#'+this._table_data.config.id) ;
            }
            return this._cont ;
        } ;

        /**
         * Event handler to be called when it's time to update the table content
         *
         * @returns {undefined}
         */
        this.on_update = function () {
        } ;
    }

    /**
     * The table viewing dialogs
     *
     * @param {Runtables_User} parent
     * @param {object} table_data
     * @returns {Runtable_DialogView}
     */
    function Runtable_DialogView (parent, table_data) {

        var _that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        Runtable_Dialog.call(this, parent, table_data) ;

        // ----------
        // Parameters
        // ----------

        ;

        // --------------------
        // Own data and methods
        // --------------------

        // -- The element for displaying the last modification info (time and a user) for the table --

        this._info = null ;
        this._get_info = function () {
            if (!this._info) {
                this._info = this._get_cont().find('#info') ;
            }
            return this._info ;
        } ;
        this._set_info = function (html) { this._get_info().html(html) ; } ;

        // -- The element for displaying the last time the table was loaded --

        this._updated = null ;
        this._get_updated = function () {
            if (!this._updated) {
                this._updated = this._get_cont().find('#updated') ;
            }
            return this._updated ;
        } ;
        this._set_updated = function (html) { this._get_updated().html(html) ; } ;

        // -- The "smart" table object --

        this._table = null ;
        this._get_table = function ()  {
            if (!this._table) {

                var hdr = ['RUN'] ;

                var column_mode = this._get_cont().find('select.display-trigger[name="column_mode"]').val() ;

                var coldef = this._table_data.config.coldef ;
                for (var i in coldef) {
                    var col = coldef[i] ;
//                    if (col.type.substring(0, 5) === 'EPICS') {
                        switch (column_mode) {
                            case 'descr'        : hdr.push(col.name) ; break ;
                            case 'pv'           : hdr.push(col.source) ; break ;
                            case 'descr_and_pv' : hdr.push(col.name + '&nbsp; &nbsp;('+col.source+')') ; break ;
                        }
//                    } else {
//                        hdr.push(col.name) ;
//                    }
                }

                var num_hdr_rows = 2 ,
                    max_hdr_rows = 5 ;

                this._table = new SmartTable (
                    hdr ,
                    [] ,            // row data will be loaded later
                    num_hdr_rows ,
                    max_hdr_rows
                ) ;
                this._table.display(this._get_cont().find('div#table')) ;
            }
            return this._table ;
        } ;
        this._re_create_table = function ()  {
            this._table = null ;
            this._table = this._get_table() ;
        } ;

        /**
         * Reimplement the event handler from the base class.
         *
         * @returns {undefined}
         */
        this.on_update = function () {
            //if (!this._is_shown) return ;
            this._load() ;
        } ;

        this._is_shown = false ;    // checked by show() and completed by display().

        /**
         * Show the dialog
         *
         * @returns {undefined}
         */
        this.show = function () {

            if (this._is_shown) return ;

            this._get_cont_hdr().html(this._table_data.config.name) ;

            var html =
'    <div id="ctrl">' +
'      <div class="group">' +
'        <span class="label">Runs</span>' +
'        <select class="update-trigger" name="runs" >' +
'          <option value="20" >last 20</option>' +
'          <option value="100">last 100</option>' +
'          <option value="200">last 200</option>' +
'          <option value="range">specific range</option>' +
'          <option value="all">all</option>' +
'        </select>' +
'      </div>' +
'      <div class="group">' +
'        <span class="label">First</span>' +
'        <input class="update-trigger" type="text" name="first" value="-20" size=8 disabled />' +
'        <span class="label">Last</span>' +
'        <input class="update-trigger" type="text" name="last"  value="" size=8 disabled />' +
'      </div>' +
'      <div class="buttons">' +
'        <button class="control-button"               name="reset"  title="reset the form"             >RESET FORM</button>' +
'        <button class="control-button update-button" name="update" title="update and display results" ><img src="../webfwk/img/Update.png" /></button>' +
'      </div>' ;
            if (this._parent.access_list.runtables.edit) html +=
'      <div class="buttons">' +
'        <button class="control-button owner-restricted edit"   name="edit"   title="edit table configuration: add/remove/remove/rearrange columns, rename the table, etc." >RE-CONFIGURE</button>' +
'        <button class="control-button owner-restricted delete" name="delete" title="delete the table" >DELETE</button>' +
'      </div>' ;
            html +=
'      <div style="clear:both;"></div>' +
'      <div class="group">' +
'        <textarea name="readonly_descr"></textarea>' +
'      </div>' +
'      <div style="clear:both;"></div>' +
'    </div>' +
'    <div id="body">' +
'      <div class="info" id="info"    style="float:left;">&nbsp;</div>' +
'      <div class="info" id="updated" style="float:right;">&nbsp;</div>' +
'      <div style="clear:both;"></div>' +
'      <div id="table_ctrl">' +
'        <span>Display columns as</span>' +
'        <select class="display-trigger" name="column_mode">' +
'          <option value="descr"        >description</option>' +
'          <option value="pv"           >name</option>' +
'          <option value="descr_and_pv" >description (name)</option>' +
'        </select>' +
'      </div>' +
'      <div id="table" class="table" ></div>' +
'    </div>' ;
            this._get_cont().html(html) ;

            var descr_elem = this._get_cont().find('textarea[name="readonly_descr"]') ;
            descr_elem.val(this._table_data.config.descr) ;
            descr_elem.attr('disabled', 'disabled') ;

            this._get_cont().find('button.control-button').button().click(function () {
                var op = this.name ;
                switch (op) {
                    case 'update' : _that._load() ; break ;
                    case 'reset'  : _that._reset() ; break ;
                    case 'edit'   : _that._parent._edit_table(_that._table_data.config.id) ; break ;
                    case 'delete' : _that._delete() ; break ;
                }
            }) ;
            this._get_cont().find('select.update-trigger').change(function () {
                var range = $(this).val() ;
                if (range === 'range') {
                    _that._get_cont().find('input.update-trigger[name="first"]').removeAttr('disabled') ;
                    _that._get_cont().find('input.update-trigger[name="last"]') .removeAttr('disabled') ;
                } else {
                    _that._get_cont().find('input.update-trigger[name="first"]').attr('disabled', 'disabled') ;
                    _that._get_cont().find('input.update-trigger[name="last"]') .attr('disabled', 'disabled') ;
                    switch (range) {
                        case '20' :
                        case '100' :
                        case '200' :
                            _that._get_cont().find('input[name="first"]').val(-parseInt(range)) ;
                            _that._get_cont().find('input[name="last"]').val('') ;
                            break ;
                        case 'all' :
                            _that._get_cont().find('input[name="first"]').val('') ;
                            _that._get_cont().find('input[name="last"]').val('') ;
                            break ;
                    }
                }
            }) ;
            this._get_cont().find('.update-trigger').change(function () {
                _that._load() ;
            }) ;
            this._get_cont().find('select.display-trigger').change(function () {
                 console.log('display: '+$(this).val()) ;
                 _that._re_create_table() ;
                 _that._display() ;
             }) ;

            this._load() ;
        } ;
        function _value_display_trait_for (col_type) {
            return col_type === 'DAQ Detectors' ?
                function (val) { return val ? '<div style="width:100%; text-align:center; font-size:14px; color:red;">&diams;</div>' : '' ; } :
                function (val) { return val ? val : '' ; } ;
        }
        this._display = function () {

            var title = 'show the run in the e-Log Search panel within the current Portal' ;

            var rows = [] ;
            for (var run in this._table_data.runs) {
                var row = [] ;
                row.push(
                    '<a href="javascript:global_elog_search_run_by_num('+run+',true);" title="'+title+'" class="link">'+run+'</a>'
                ) ;
                var param2value = this._table_data.runs[run] ;
                var coldef = this._table_data.config.coldef ;
                for (var i in coldef) {
                    var col = coldef[i] ;
                    var name  = col.name ;

                    var value = name in param2value ? param2value[name] : '' ;

                    // TODO: replace with lazy initialization using JQuery to
                    //       prevent markup screwup due to special symbols.

                    if (col.is_editable && this._parent.access_list.runtables.edit) {
                        var run_id = this._table_data.run2id[run] ;
                        row.push('<input class="editable" coldef_id="'+col.id+'" run_id="'+run_id+'" type="text" value="'+value+'" />') ;
                    } else {
                        var value_display = _value_display_trait_for(col.type) ;
                        row.push(value_display(value)) ;
                    }
                }
                rows.push(row) ;
            }
            rows.reverse() ;

            this._get_table().load(rows) ;

            // Set up event handlers for updating cells

            this._get_cont().find('div#table').find('input.editable').change(function () {

                var elem = $(this) ;
                elem.addClass('modified') ;

                var coldef_id = parseInt(elem.attr('coldef_id')) ;
                var run_id    = parseInt(elem.attr('run_id')) ;
                var value     = elem.val() ;

                _that._set_updated('Saving...') ;

                _that._update_cell (
                    run_id, coldef_id, value,
                    function (data) {
                        elem.removeClass('modified') ;
                        _that._table_data.config.modified_time = data.info.modified_time ;
                        _that._table_data.config.modified_uid  = data.info.modified_uid ;
                        _that._set_info   ('Re-configured: <b>'+data.info.modified_time+'</b> by: <b>'+data.info.modified_uid+'</b>') ;
                        _that._set_updated('Updated: <b>'+data.updated+'</b>') ;
                    }
                ) ;
            }) ;

            this._is_shown = true ;
        } ;

        this._update_cell = function (run_id, coldef_id, value, on_success) {
            var cells = [{
                run_id:    run_id ,
                coldef_id: coldef_id ,
                value:     value
            }] ;
            var params = {
                exper_id: this._parent.experiment.id ,
                table_id: this._table_data.config.id ,
                cells:    JSON.stringify(cells)
            } ;
            Fwk.web_service_POST (
                '../portal/ws/runtable_user_cell_save.php' ,
                params ,
                on_success
            ) ;
        } ;

        this._load = function () {

            this._set_updated('Loading...') ;

            var params = {
                exper_id: this._parent.experiment.id ,
                table_id: this._table_data.config.id
            } ;
            var range = this._get_cont().find('select[name="runs"]').val() ;
            switch (range) {
                case '20' :
                case '100' :
                case '200' :
                    params.from_run    = -parseInt(range) ;
                    params.through_run = 0 ;
                    break ;
                case 'range' :
                    params.from_run    = parseInt(this._get_cont().find('input[name="first"]').val()) ;
                    params.through_run = parseInt(this._get_cont().find('input[name="last"]').val()) ;
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
                    _that._table_data.runs   = data.runs ;
                    _that._table_data.run2id = data.run2id ;
                    _that._table_data.config.modified_time = data.info.modified_time ;
                    _that._table_data.config.modified_uid  = data.info.modified_uid ;
                    _that._set_info   ('Re-configured: <b>'+data.info.modified_time+'</b> by: <b>'+data.info.modified_uid+'</b>') ;
                    _that._set_updated('Updated: <b>'+data.updated+'</b>') ;
                    _that._display() ;
                }
            ) ;
        } ;

        this._reset = function () {
            this._get_cont().find('.update-trigger').val('') ;
            this._load() ;
        } ;
        this._delete = function () {

            var button_delete = this._get_cont().find('button[name="delete"]').button() ;
            button_delete.button('disable') ;

            Fwk.ask_yes_no (
                'Confirm Destructive Operation' ,
                'Would you really like to delete the table? '+
                'Note that after pressing <b>Yes</b> the information stored in the user-editable columns will be destroyed in the database.' ,
                function () {
                    var params = {
                        exper_id: _that._parent.experiment.id ,
                        table_id: _that._table_data.config.id
                    } ;
                    Fwk.web_service_GET (
                        '../portal/ws/runtable_user_delete.php' ,
                        params ,
                        function () {
                            _that._parent._delete_table(_that._table_data.config.id) ;
                            button_delete.button('enable') ;
                        } ,
                        function (msg) {
                            Fwk.report_error(msg) ;
                            button_delete.button('enable') ;
                        }
                    ) ;
                }
            ) ;
        } ;
    }
    Class.define_class (Runtable_DialogView, Runtable_Dialog, {}, {}) ;

    /**
     * The table editing dialogs
     *
     * NOTE: The very same class is also used to create new tables which
     * does not exist yet in the database. This mode is triggered by the last
     * parameter of the class's constructor.
     *
     * @param {Runtables_User} parent
     * @param {object} table_data
     * @returns {Runtable_DialogEdit}
     */
    function Runtable_DialogEdit (parent, table_data, new_table_mode) {

        var _that = this ;

        // -----------------------------------------
        // Allways call the base class's constructor
        // -----------------------------------------

        Runtable_Dialog.call(this, parent, table_data) ;

        // ----------
        // Parameters
        // ----------

        this._new_table_mode = new_table_mode ? true : false ;

        // --------------------
        // Own data and methods
        // -------------------- 

        // -- ?

        this._info = null ;
        this._get_info = function () {
            if (!this._info) {
                this._info = this._get_cont().find('div#info') ;
            }
            return this._info ;
        } ;

        // -- ?

        this._columns = null ;
        this._get_columns = function () {
            if (!this._columns) {
                this._columns = this._get_cont().find('table.columns') ;
            }
            return this._columns ;
        } ;

        // -- Operations on the button for saving modifications --

        this._button_save = null ;
        this._set_button_save = function (state) {
            if (!this._button_save) {
                this._button_save = this._get_cont().find('#ctrl').find('button[name="save"]').button() ;
            }
            this._button_save.button(state) ;
        }

        /**
         * Show the dialog
         *
         * @returns {undefined}
         */
        this.show = function () {

            this._get_cont_hdr().html (
                this._new_table_mode ? 
                '<span style="color:red;">CREATE NEW TABLE</span>' : 
                this._table_data.config.name+': <span style="color:red;">editing</span>'
            ) ;

            var table_id = this._table_data.config.id ;

            var html =
'<div id="ctrl">' +
'  <div class="group">'+(this._new_table_mode ?
'    <p><b>INSTRUCTIONS:</b> Fill in a desired table name and configure the initial columns before hitting' +
'       the <b>SAVE</b> button. Note that more columns can be added later using the table manager' +
'       dialog when viewing the table. If no columns are provided when creating the table then the table' +
'       will initially have just one - for run numbers. Columns can also be removed later.' +
'       Each source is composed of a <span style="font-style:italic;">description</span>' +
'       followed by its formal <span style="font-style:italic;">name</span> included in the round parentheses.' +
'       The current state of the global selector instructs the editor which part of' +
'       the sources to pull into column names. Feel free to change the selector at any moment.' +
'       It will not affect previously formed columns.' +
'    </p>' :
'    <p><b>IMPORTANT:</b> Removing columns of the <b>Editable</b> type or changing their type will' +
'       result in loosing all relevant information from the database after you hit the <b>Save</b> button.') +
'    </p>' +
'    <div>' +
'      <span class="label">Table name</span>' +
'      <input type="text" size="50" name="name" />' +
'    </div>' +
'    <div>' +
'      <textarea name="descr" title="an optional description for the table" /></>' +
'    </div>' +
'  </div>' +
'  <div class="buttons">'+(this._new_table_mode ?
'    <button name="reset"  class="control-button" title="reset the table creation form" >RESET FORM</button>' +
'    <button name="create" class="control-button important-button" title="create the table" >CREATE</button>' :
'    <button name="save"   class="control-button" title="permanently save the modifications \nand switch to the viewing mode" >SAVE</button>' +
'    <button name="cancel" class="control-button" title="cancel the table editing dialog \nand switch to the viewing mode" >CANCEL</button>') +
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
'      <div id="table_ctrl">' +
'        <span>The editor will pull</span>' +
'        <select class="display-trigger" name="column_mode">' +
'          <option value="descr" >description</option>' +
'          <option value="pv"    >name</option>' +
'        </select>' +
'        <span>from sources</span>' +
'      </div>' +
'      <table class="columns" border="0" cellspacing="0">' +
'        <thead>' +
'          <tr>' +
'            <td>&nbsp;</td>' +
'            <td>Column Name</td>' +
'            <td>Category</td>' +
'            <td>Data Source <span style="font-weight:normal">description (name)</span></td>' +
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
            this._get_cont().html(html) ;
            this._get_cont().find('#ctrl').find('input[name="name"]').val(this._table_data.config.name) ;
            this._get_cont().find('#ctrl').find('textarea[name="descr"]').val(this._table_data.config.descr) ;

            this._get_cont().find('#'+table_id+'-tabs').tabs() ;

            this._get_cont().find('.control-button').button().click(function () {
                var op = this.name ;
                switch (op) {
                    case 'create': _that._create() ; break ;
                    case 'reset' : _that._reset() ; break ;
                    case 'cancel': _that._parent._view_table(table_id) ; break ;
                    case 'save'  : _that._save() ; break ;
                }
            }) ;

            this._set_button_save('disable') ;

            this._load_column_types(function (dict) {
                _that._table_data.dict = dict ;
                _that._init_columns() ;
                _that._set_button_save('enable') ;
                _that._get_cont().find('select.display-trigger[name="column_mode"]').off().change(function () {
                    _that._adjust_column_types() ;
                 }) ;
            }) ;
        } ;

        this._load_column_types = function (when_done) {
            Fwk.web_service_GET (
                '../portal/ws/runtable_user_dict_get.php' ,
                {exper_id: this._parent.experiment.id} ,
                function (data) {
                    when_done(data.types) ;
                }
            ) ;
        } ;

        this._init_columns = function () {

            var row = 0 ;

            var tbody = this._get_columns().find('tbody') ;

            var coldef = this._table_data.config.coldef ;
            for (var i in coldef) {

                var col = coldef[i] ;

                var tr_last = tbody.find('tr.coldef:last') ;
                if (tr_last.length) tr_last.after(this._column_paceholder(row, col)) ;
                else                tbody.html(this._column_paceholder(row, col)) ;

                this._install_column_types(row, col.type) ;

                var tr = tbody.find('tr.coldef:last') ;
                tr.removeClass('spare') ;
                tr.find('input[name="column"]') .val(col.name) ;
                tr.find('select[name="type"]')  .val(col.type) ;
                tr.find('select[name="source"]').val(col.source) ;

                row++ ;
            }
            this._update_events() ;             // make sure all event handlers are activated
            this._update_info() ;
            this._add_column_paceholder_if_needed() ;
        } ;

        this._save = function () {

            var name = this._get_cont().find('#ctrl').find('input[name="name"]').val() ;
            if (name === '') {
                Fwk.report_error('Please, fill in the table name field and try again!') ;
                return ;
            }

            var coldef = [] ;
            this._get_columns().find('tr.coldef').each(function () {
                var tr = $(this) ;
                if (tr.hasClass('spare')) return ;
                coldef.push ({
                    name:     tr.find('input[name="column"]').val() ,
                    type:     tr.find('select[name="type"]').val() ,
                    source:   tr.find('select[name="source"]').val() ,
                    id:       parseInt(tr.attr('cid')) ,
                    position: parseInt(tr.attr('pos'))
                }) ;
            }) ;

            this._set_button_save('disable') ;

            var params = {
                exper_id: this._parent.experiment.id ,
                id:       this._table_data.config.id ,
                name:     name ,
                descr:    this._get_cont().find('#ctrl').find('textarea[name="descr"]').val() ,
                coldef:   JSON.stringify(coldef)
            } ;
            Fwk.web_service_POST (
                '../portal/ws/runtable_user_reconfigure.php' ,
                params ,
                function (data) {
                    _that._table_data = data.table_data ;
                    _that._parent._view_table(_that._table_data.config.id) ;
                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                    _that._set_button_save('enable') ;
                }
            ) ;
        } ;

         this._create = function () {

            var name = this._get_cont().find('input[name="name"]').val() ;
            if (name === '') {
                Fwk.report_error('Please, fill in the table name and try again!') ;
                return ;
            }

            var coldef = [] ;
            this._get_columns().find('tr.coldef').each(function () {
                var tr = $(this) ;
                if (tr.hasClass('spare')) return ;
                coldef.push ({
                    name:     tr.find('input[name="column"]').val() ,
                    type:     tr.find('select[name="type"]').val() ,
                    source:   tr.find('select[name="source"]').val() ,
                    position: tr.attr('pos')
                }) ;
            }) ;

            var button_create = this._get_cont().find('button[name="create"]').button() ;
            button_create.button('disable') ;

            var params = {
                exper_id: this._parent.experiment.id ,
                name:     name ,
                descr:    this._get_cont().find('textarea[name="descr"]').val() ,
                coldef:   JSON.stringify(coldef)
            } ;
            Fwk.web_service_POST (
                '../portal/ws/runtable_user_create.php' ,
                params ,
                function (data) {
                    _that._parent._add_table(data.table_data) ;

                } ,
                function (msg) {
                    Fwk.report_error(msg) ;
                    button_create.button('enable') ;
                }
            ) ;
        } ;

        this._reset = function () {
            this._get_cont().find('input[name="name"]').val('') ;
            this._get_columns().find('tr.coldef').each(function () {
                var tr = $(this) ;
                var row = parseInt(tr.attr('row')) ;
                if (row) tr.remove() ;
            }) ;
            this._clear_column(0) ;
        } ;

        this._clear_column = function (row) {
            var tr = this._get_columns().find('tr.coldef[row="'+row+'"]') ;
            if (tr.length) {
                tr.find('input[name="column"]').val('') ;
                tr.find('select[name="type"]').val('') ;
                tr.find('select[name="source"]').html('').attr('disabled', 'disabled') ;
                tr.addClass('spare') ;
                this._update_info() ;

            }
            this._add_column_paceholder_if_needed() ;
        } ;

        this._move_column2left = function (row) {
            var tr = this._get_columns().find('tr.coldef[row="'+row+'"]') ;
            if (tr.length) {
                var tr_prev = tr.prev() ;
                if (tr_prev.length) {

                    var pos      = tr     .attr('pos') ;
                    var pos_prev = tr_prev.attr('pos') ;

                    tr     .attr('pos', pos_prev) ;
                    tr_prev.attr('pos', pos) ;

                    tr.after(tr_prev) ;

                    this._update_events() ;
                    this._update_info() ;
                }
            }
        } ;

        this._move_column2right = function (row) {
            var tr = this._get_columns().find('tr.coldef[row="'+row+'"]') ;
            if (tr.length) {
                var tr_next = tr.next() ;
                if (tr_next.length) {

                    var pos      = tr     .attr('pos') ;
                    var pos_next = tr_next.attr('pos') ;

                    tr     .attr('pos', pos_next) ;
                    tr_next.attr('pos', pos) ;

                    tr_next.after(tr) ;

                    this._update_events() ;
                    this._update_info() ;
                }
            }
        } ;

        this._column_paceholder = function (row, col) {
            var html =
'<tr class="coldef spare" row="'+row+'" cid="'+col.id+'" pos="'+col.position+'">' +
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
        this._add_column_paceholder_if_needed = function () {

            var tr_spare = this._get_columns().find('tr.spare') ;
            if (tr_spare.length) return ;

            var tr_last = this._get_columns().find('tr.coldef:last') ;
            var row = undefined ;
            if (tr_last.length) {
                row = parseInt(tr_last.attr('row')) + 1 ;
                var col = {
                    id: 0 ,
                    position: this._get_columns().find('tr.coldef').length
                } ;
                tr_last.after(this._column_paceholder(row, col)) ;
            } else {
                row = 0 ;
                var col = {
                    id: 0 ,
                    position: 0
                } ;
                this._get_columns().find('tbody').html(this._column_paceholder(row, col)) ;
            }
            this._update_events() ;             // make sure all event handlers are activated
            this._install_column_types(row) ;   // initialize selectors in the row
            this._update_info() ;
        } ;

        this._update_events = function () {

            // ATTENTION: note all previously set event handlers are removed to avoid
            //            multiple execution!

            this._get_columns().find('button[name="clear"]').button().off().click(function () {
                var row =  parseInt($(this).closest('tr').attr('row')) ;
                _that._clear_column(row) ;
            }) ;
            this._get_columns().find('button[name="left"]').button().off().click(function () {
                var row =  parseInt($(this).closest('tr').attr('row')) ;
                _that._move_column2left(row) ;
            }) ;
            this._get_columns().find('button[name="right"]').button().off().click(function () {
                var row =  parseInt($(this).closest('tr').attr('row')) ;
                _that._move_column2right(row) ;
            }) ;
            this._get_columns().find('input[name="column"]').off().change(function () {
                var elem = $(this) ;
                var tr = elem.closest('tr') ;
                var column = elem.val() ;
                if (column === '')
                    tr.addClass('spare') ;
                else {
                    tr.removeClass('spare') ;
                    _that._add_column_paceholder_if_needed() ;
                }
                _that._update_info() ;
            }) ;
            this._get_columns().find('select[name="type"]').off().change(function () {
                var elem = $(this) ;
                var tr = elem.closest('tr') ;
                var row =  parseInt(tr.attr('row')) ;
                var select_type = elem.val() ;
                var cid =  parseInt(tr.attr('cid')) ;
                if (cid) {
                    var coldef = _that._table_data.config.coldef ;
                    for (var i in coldef) {
                        var col = coldef[i] ;
                        if ((col.id == cid) && col.is_editable && (col.type != select_type)) {
                            Fwk.ask_yes_no (
                                'Changing Column Type' ,
                                'Changing a type of the Editable column may result in the permanent loss of all relevant data entered by users' ,
                                function () {
                                    // -- Keep the modifications --
                                    if (select_type === '') _that._install_column_types(row) ;
                                    else                    _that._install_column_types(row, select_type) ;
                                    _that._add_column_paceholder_if_needed() ;
                                } ,
                                function () {
                                    // -- Revert back to the previous type --
                                    _that._install_column_types(row, col.type) ;
                                    _that._add_column_paceholder_if_needed() ;
                                }
                            ) ;
                            return ;
                        }
                    }
                }
                // -- Keep the modifications --
                if (select_type === '') _that._install_column_types(row) ;
                else                    _that._install_column_types(row, select_type) ;
                _that._add_column_paceholder_if_needed() ;
            }) ;
            this._get_columns().find('select[name="source"]').off().change(function () {
                var elem = $(this) ;
                var tr = elem.closest('tr') ;
                var select_source = elem.val() ;
                //var select_descr  = elem.find('option[value="'+select_source+'"]').text() ;
                var select_descr  = elem.find('option[value="'+select_source+'"]').attr('name') ;
                tr.find('input[name="column"]').val(select_descr) ;
                if (select_source === '') {
                    tr.addClass('spare') ;
                } else {
                    tr.removeClass('spare') ;
                    _that._add_column_paceholder_if_needed() ;
                }
                _that._update_info() ;
            }) ;
        } ;
        this._update_info = function () {
            var tr = this._get_columns().find('tr.coldef') ;
            var named = 0 ;
            var editable = 0 ;
            tr.each(function () {
                var elem = $(this) ;
                if (!elem.hasClass('spare')) named++ ;
                if (elem.find('select[name="type"]').val() === 'Editable') editable++ ;
            }) ;
            var html = '<b>'+named+'</b> columns [ EDITABLE: <b>'+editable+'</b> ]' ;
            this._get_info().html(html) ;

            this._update_preview() ;
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
        this._install_column_types = function (row, select_type, select_source) {
            var column_mode = this._get_cont().find('select.display-trigger[name="column_mode"]').val() ;
            var tr = this._get_columns().find(row === undefined ? 'tr.coldef' : 'tr.coldef[row="'+row+'"]') ;
            if (tr.length) {
                var html = '<option value="">&lt;select&gt;</option>' ;
                for (var type in this._table_data.dict) {
                    html += '<option>'+type+'</option>' ;
                }
                var type_elem = tr.find('select[name="type"]') ;
                type_elem.html(html) ;
                var source_elem = tr.find('select[name="source"]') ;
                if (select_type !== undefined) {
                    type_elem.val(select_type) ;
                    if (select_type in this._table_data.dict) {
                        var html = '<option value="">&lt;select&gt;</option>' ;
                        for (var i in this._table_data.dict[select_type]) {
                            var p = this._table_data.dict[select_type][i] ;
                            var source = p.name ;
                            var descr  = p.descr ;
//                            if (select_type.substr(0,5) === 'EPICS') {
                                var opt = descr+' &nbsp;&nbsp; ('+source+')' ;
                                switch (column_mode) {
                                    case 'descr'        : html += '<option value="'+source+'" name="'+descr+'"  >'+opt+'</option>' ; break ;
                                    case 'pv'           : html += '<option value="'+source+'" name="'+source+'" >'+opt+'</option>' ; break ;
                                }
//                            } else {
//                                html += '<option value="'+source+'" name="'+descr+'" >'+descr+'</option>' ;
//                            }
                        }
                        source_elem.html(html) ;
                        source_elem.removeAttr('disabled') ;
                        if (select_source !== undefined) {
                            source_elem.val(select_source) ;
                        }
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

        /**
         * Adjust selectors for EPICS types to display PV names, descriptiosn or both
         * as per the current configuration of the UI.
         *
         * @returns {undefined}
         */
        this._adjust_column_types = function () {

            this._get_columns().find('tr.coldef').each(function () {
                var tr = $(this) ;
                var row = tr.attr('row') ;

                var type = tr.find('select[name="type"]').val() ;
//                if ((type !== '') && (type.substring(0, 5) === 'EPICS')) {

                    var source = tr.find('select[name="source"]').val() ;
                    if (source !== '') _that._install_column_types(row, type, source) ;
                    else               _that._install_column_types(row, type) ;
//                }
            }) ;
        } ;

        this._update_preview = function ()  {

            var table_cont = this._get_cont().find('div#table.table') ;

            var hdr  = ['RUN'] ;
            var rows = [] ;

            var trs = this._get_columns().find('tr.coldef') ;
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
                                    row.push('<input type="text" value="" >') ;
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
            var num_hdr_rows = 2 ,
                max_hdr_rows = 5 ;

            var table = new SmartTable (
                hdr ,
                rows ,
                num_hdr_rows ,
                max_hdr_rows
            ) ;
            table.display(table_cont) ;
        } ;
    }
    Class.define_class (Runtable_DialogEdit, Runtable_Dialog, {}, {}) ;

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
            this._init() ;
        } ;

        this.on_deactivate = function() {
            ;
        } ;

        this._update_ival_sec = 10 ;
        this._prev_update_sec = 0 ;

        this.on_update = function () {
            if (!this.active) return ;
            if (!this._is_initialized) return ;
            var now_sec = Fwk.now().sec ;
            if (this._prev_update_sec + this._update_ival_sec < now_sec) {
                this._update() ;
                this._prev_update_sec = now_sec ;
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

        this._table_dialog = [] ;     // table dialogs indexed by table identifiers

        this._last_runnum = 0 ;

        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this.container.html('<div id="runtables-user"></div>') ;
            this._wa = this.container.find('div#runtables-user') ;
            this._wa.html('Loading...') ;

            if (!this.access_list.runtables.read) {
                this._wa.html(this.access_list.no_page_access_html) ;
                return ;
            }

            this._load_tables(function (data) {
                _that._display_tables(data.table_data) ;
            }) ;
        } ;

        /**
         * Make updates to the page content if needed
         * 
         * Here is what gets updated and when:
         * - if there are new tables then add them
         * - if tables are gone then remove them
         * - go through all table viewers (don't touch editors) dialogs which
         *   have been modified and then, depenidng on what's modified:
         *   - if the table configurations (name, description, coldef) were
         *     modified then recreate the viewing dialogs
         *   - if the timestamps were modified (which is the case for more runs added
         *     or modified cells) then call the 'on_update() event handlers on
         *     those tables.
         *
         * @returns {undefined}
         */
        this._update = function () {
            this._load_tables(function (data) {

                // -- add tables which are missing --

                for (var id in data.table_data) {
                    if (id in _that._table_dialog) continue ;

                    _that._prepend_tab_panel(id, '', '') ;
                    _that._table_dialog[id] = new Runtable_DialogView(_that, data.table_data[id]) ;
                    _that._table_dialog[id].show() ;
                }

                // -- remove tables which are gone for good --

                for (var id in _that._table_dialog) {
                    if (id in data.table_data) continue ;
                    if (id === 'add') continue ;

                    _that._delete_table(id) ;
                }

                // -- push updates (if any) to the table viewer dialogs --

                for (var id in _that._table_dialog) {

                    if (id === 'add') continue ;

                    var dialog = _that._table_dialog[id] ;
                    if (!dialog instanceof Runtable_DialogView) continue ;

                    if (dialog._table_data.config.modified_time !== data.table_data[id].config.modified_time) {

                        // -- modified configuration --

                        var modified_conf =
                            dialog._table_data.config.name          !== data.table_data[id].config.name  ||
                            dialog._table_data.config.descr         !== data.table_data[id].config.descr ||
                            dialog._table_data.config.coldef.length !== data.table_data[id].config.coldef.length ;

                        var coldef_old = dialog._table_data.config.coldef ;
                        var coldef_new = data.table_data[id].config.coldef ;

                        if (!modified_conf) {
                            for (var i in coldef_old) {
                                if (i in coldef_new ) continue ;
                                modified_conf = true ;
                                break ;
                            }
                        }
                        if (!modified_conf) {
                            for (var i in coldef_new) {
                                if (i in coldef_old ) continue ;
                                modified_conf = true ;
                                break ;
                            }
                        }
                        if (!modified_conf) {
                            for (var i in coldef_new) {
                                var col_new = coldef_new[i] ;
                                var col_old = coldef_old[i] ;
                                if (col_new.id     === col_old.id   &&
                                    col_new.name   === col_old.name &&
                                    col_new.type   === col_old.type &&
                                    col_new.source === col_old.source) continue ;
                                modified_conf = true ;
                                break ;
                            }
                        }
                        if (modified_conf) {
                            _that._table_dialog[id] = new Runtable_DialogView(_that, data.table_data[id]) ;
                            _that._table_dialog[id].show() ;
                        }

                        dialog.on_update() ;

                    } else {

                        // --  check if the payload has been modified --

                        // TODO: In theory it would be nice to let the table to decide
                        //       if it needs to update itself by forwarding the update signal
                        //       to the table. But for now let's disable this and only track
                        //       differences in run numbers.
                        // 
                        // dialog.on_update() ;


                        if (_that._last_runnum != data.last_runnum) {
                            _that._table_dialog[id] = new Runtable_DialogView(_that, data.table_data[id]) ;
                            _that._table_dialog[id].show() ;
                        }

                    }
                }
                _that._last_runnum = data.last_runnum ;
            }) ;
        } ;

        this._load_tables = function (on_success) {
            Fwk.web_service_GET (
                '../portal/ws/runtable_user_tables.php' ,
                {exper_id: this.experiment.id} ,
                function (data) {
                    on_success(data) ;
                } ,
                function (msg) {
                    _that._wa.html(msg) ;
                }
            ) ;
        } ;

        this._display_tables = function (table_data) {

            var html =
'<div id="tabs">' +
'  <ul></ul>' +
'</div>' ;

            this._wa.html(html) ;
            this._tabs = this._wa.children('#tabs').tabs() ;

            this._table_dialog = [] ;
            for (var i in table_data) {
                this._append_tab_panel(table_data[i].config.id, '', '') ;
                this._table_dialog[table_data[i].config.id] = new Runtable_DialogView(this, table_data[i]) ;
                this._table_dialog[table_data[i].config.id].show() ;
            }
            if (this.access_list.runtables.edit) {
                this._append_tab_panel('add', 'Add new table', '<span class="ui-icon ui-icon-plus"></span>') ;
                this._set_new_table_dialog() ;
            }
        } ;

        /**
         * 
         * @param {type} id
         * @param {string} title
         * @param {type} a
         * @param {type} prepend
         * @returns {undefined}
         */
        this._new_tab_panel = function (id, title, a, prepend) {
            var li_html  = '<li><a href="#'+id+'" title="'+title+'" >'+a+'</a></li>' ;
            var div_html = '<div id="'+id+'"></div>' ;
            if (prepend) this._tabs.children('.ui-tabs-nav').prepend(li_html) ;
            else         this._tabs.children('.ui-tabs-nav').append (li_html) ;
            this._tabs.append(div_html) ;
            this._tabs.tabs("refresh") ;
            this._tabs.tabs("select", 0) ;
            this._tabs.tabs("option", "active", 0) ;
            this._set_new_table_dialog() ;
        } ;
        this._prepend_tab_panel = function (id, title, a) { this._new_tab_panel(id, title, a, true) ; } ;
        this._append_tab_panel  = function (id, title, a) { this._new_tab_panel(id, title, a, false) ; } ;

        /**
         * Reinitialize the table creator dialog within existing tab panel
         * 
         * This method will create a fake table data object to be passed as a parameter
         * of teh dialog.
         *
         * @returns {undefined}
         */
        this._set_new_table_dialog = function () {

            var new_table_data = {
                config: {
                    id: 'add' ,
                    name: '' ,
                    coldef: []
                }
            }
            var new_table_dialog = new Runtable_DialogEdit(this, new_table_data, true) ;
            this._table_dialog['add'] = new_table_dialog ;
            new_table_dialog.show() ;
        } ;

        /**
         * Toggle the table dialog to the specified mode
         * 
         * Note, that thsi implementation will be always reusing the table data
         * when switching between modes.
         *
         * @param {number} id
         * @param {boolean} edit_mode
         * @returns {undefined}
         */
        this._toggle_dialog_mode = function (id, edit_mode) {
            var dialog_class = edit_mode ? Runtable_DialogEdit : Runtable_DialogView ;
            this._table_dialog[id] = new dialog_class(this, this._table_dialog[id]._table_data) ;
            this._table_dialog[id].show() ;
        } ;
        this._edit_table = function (id) { this._toggle_dialog_mode(id, true) ; }
        this._view_table = function (id) { this._toggle_dialog_mode(id, false) ; }

        this._add_table = function (table_data) {
            this._prepend_tab_panel(table_data.config.id, '', '') ;
            this._table_dialog[table_data.config.id] = new Runtable_DialogView(this, table_data) ;
            this._table_dialog[table_data.config.id].show() ;
            this._set_new_table_dialog() ;
        } ;

        /**
         * Remove the corresponding tab and the corresponding dialog object
         *
         * @param {number} id
         * @returns {undefined}
         */
        this._delete_table = function (id) {
            this._tabs.children(".ui-tabs-nav").find('a[href="#'+id+'"]').parent().remove() ;
            this._tabs.children('div#'+id) ;
            this._tabs.tabs("refresh") ;
            delete this._table_dialog[id] ;
        } ;
    }
    Class.define_class (Runtables_User, FwkApplication, {}, {}) ;

    return Runtables_User ;
}) ;
