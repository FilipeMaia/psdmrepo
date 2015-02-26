define ([
    'webfwk/CSSLoader', 'webfwk/Class', 'webfwk/Widget' ,
    'underscore'] ,

function (
    cssloader, Class, Widget) {

    cssloader.load('../webfwk/css/CheckTable.css') ;

    /**
     * @brief The tabular widget representing checkbox selectors in the first
     *        (mandatory) column and optional columns
     *
     * DESCRIPTION:
     * 
     *   The class respresents a table which can be statically initialized,
     *   extended, reduced. Cells in the first colum are boolean variables
     *   representing selectors for rows. This is the only mandatory column.
     *   Other columns are optional. The optional columns contain arbitrary
     *   strings.
     *   
     *   The table has the following schema:
     * 
     *     <selector name> <column name> <column> ...
     *     --------------- ------------- -------- ...
     *     boolean         string        string   ...
     *     ...             ...           ...
     *     
     *   Rows can be appended to the table at any time. Rows are deleted from
     *   the table by passing a predicate function to the 'remove' method.
     *   Where the only parameter to the predicate function is a dictionary
     *   (colum names are the keys, and their content of teh corresponding cells
     *   are the values) and representing a row to be evaluated. If the function
     *   returns 'true' then the row will be deleted. For example, the following
     *   operation will remove all columns which were not selected in the first
     *   column named 'selector':
     *
     *     CheckTable.remove_where(function (row) {
     *         return !row['selector'] ;
     *     }) ;
     *   
     *   The method will also return a list of deleted rows.
     *   
     *   The same idea applies to the 'search_where' operation.
     *
     *   INITIALIZING AND POPULATING THE TABLE:
     *
     *   * The column definition must be an array of at least 1 column (the selector
     *     column goes first). Each element of the array would have a dictionary of:
     *    
     *     { name:   <an identifier for accessing values> , // mandatory, no default allowed
     *       text:   <Column name to be displayed ,         // default: same as the name
     *       class:  <CSS classes for column cells> ,       // default: ''
     *       style:  <CSS styles for columnn cells> ,       // default: ''
     *       align:  <left|center|right> ,                  // default: left
     *       hidden: <true|false>                           // default: false
     *     }
     *
     *   * The column names starting with symbol '_' should not be used in the
     *     table definitions because the names are reserved by the class.
     *     Reserved names:
     *     
     *       _row_id : an internal identifier of a row
     *       _locked : a flag indicating of the table row should be locked
     *
     *   * Each row is represented with an object (dictionary) representing values for
     *     the table cells:
     *    
     *       { <column name> : <a value of the corresponding cell> }
     *    
     *     Cells of columns not mentioned in the object will set to the default state.
     *     Notes for the values of the (first) selector column:
     *
     *       - the default value is 'false' (not selected)
     *       - any value (if present) will be evaluated as boolean
     *
     *     The following reserved column names can be used as follow:
     *     
     *       _locked : if set to 'true' it will lock the row from being checked/unchecked
     *                 via the UI or programatically.
     *
     * @param array coldef
     * @param array rows
     * @param object options
     * @returns {CheckTable}
     */
    function CheckTable (coldef, rows, options) {

        var _that = this ;

        // -------------------------------------------
        //   Always call the c-tor of the base class
        // -------------------------------------------

        Widget.Widget.call(this) ;

        // --------------------
        //   Static functions
        // --------------------

        function _ASSERT (expression) {
            if (!expression) throw new Widget.WidgetError('CheckTable::'+arguments.callee.caller.name) ;
        }

        function _PROP (obj, prop, default_val, validator) {
            if (_.has(obj, prop)) {
                var val = obj[prop] ;
                if (validator) _ASSERT(validator(val)) ;
                return val ;
            }
            _ASSERT(!_.isUndefined(default_val)) ;
            return default_val ;
        }
        function _PROP_STRING (obj, prop, default_val) {
            return _PROP(obj, prop, default_val, _.isString) ;
        }
        function _PROP_BOOL (obj, prop, default_val) {
            return _PROP(obj, prop, default_val, _.isBoolean) ;
        }

        // ------------------------------
        //   Data members of the object
        // ------------------------------

        this._colnames = [] ;   // names of the columns in the original order
        this._coldef   = {} ;   // the definitions of the columns (column names are the keys)
        this._rows     = [] ;

        this._is_rendered = false ; // rendering is done only once

        this._tbody = null ;

        // ---------------------------------------
        //   The generator of unique identifiers
        // ---------------------------------------

        this._next_row_id = 0 ;

        this._get_row_id = function () {
            return this._next_row_id++ ;
        } ;

        // -----------------------------
        //   Digest column definitions
        // -----------------------------

        _ASSERT(_.isArray(coldef) && coldef.length) ;

        _.each(coldef, function (col) {
            _ASSERT (
                _.isObject(col) &&
                _.has(col, 'name') && _.isString(col.name) && col.name !== '') ;

            var name = _PROP_STRING(col, 'name') ;
            _ASSERT(name !== '') ;

            _that._colnames.push(name) ;
            _that._coldef[name]  = {
                is_selector: _that._colnames.length === 1 ,
                text:   _PROP_STRING(col, 'text',   name) ,   // use the name if no text is provided
                class:  _PROP_STRING(col, 'class',  '') ,
                style:  _PROP_STRING(col, 'style',  '') ,
                align:  _PROP_STRING(col, 'align',  'left') ,
                hidden: _PROP_BOOL  (col, 'hidden', false)
            } ;
        }) ;

        // ------------------------
        //   Operations with rows
        // ------------------------

        this._add_rows = function (rows) {
            if (!rows) return ;
            _ASSERT(_.isArray(rows)) ;
            _.each(rows, function (row) {
                _that._add_row(row) ;
            }) ;
        } ;
        this._add_row = function (row, position) {

            // Make a new data object representing object state

            _ASSERT (_.isObject(row)) ;
            var row2add = _.reduce(this._colnames, function (row2add, name) {
                if (_that._coldef[name].is_selector) row2add[name] = _PROP_BOOL  (row, name, false) ;
                else                                 row2add[name] = _PROP_STRING(row, name, '') ;
                return row2add ;
            } , {
                _row_id : this._get_row_id() ,
                _locked : _PROP_BOOL  (row, '_locked', false)
            }) ;

            if (_.isUndefined(position)) {
                this._rows.push(row2add) ;
            } else {
                _ASSERT (_.isNumber(position) && position >= 0) ;
                if (position >= this._rows.length) {
                    this._rows.push(row2add) ;
                } else {
                    this._rows.splice(position, 0, row2add) ;
                }
            }

            if (this._is_rendered) {
                this._display_row(row2add, position) ;
            }

        } ;
        this.append = function (rows) {
            if (!rows) return ;
            if (_.isArray(rows)) this._add_rows(rows) ;
            else                 this._add_row (rows) ;
        } ;
        this.insert_front = function (rows) {
            if (!rows) return ;
            if (_.isArray(rows)) this._add_rows(rows, 0) ;
            else                 this._add_row (rows, 0) ;
        } ;

        /**
         * @brief Remove rows which satisfy the predicate
         *
         * @param {function} predicate
         * @returns {array}
         */
        this.remove = function (predicate) {
            _ASSERT (_.isFunction(predicate)) ;
            var rows2remove = _.filter(this._rows, predicate) ;
            _.each(rows2remove, function (row) {
                _that._undisplay_row(row) ;
                for (var idx in _that._rows) {
                    if (_that._rows[idx]._row_id === row._row_id) {
                        _that._rows.splice(idx, 1) ;
                        break ;
                    }
                }
            }) ;
            return rows2remove ;
        } ;
        this.remove_all = function () { return this.remove(function () { return true ; }) } ;

        /**
         * @brief Return rows which satisfy the predicate
         *
         * @param {function} predicate
         * @returns {array}
         */
        this.find = function (predicate) {
            _ASSERT (_.isFunction(predicate)) ;
            return _.filter(this._rows, predicate) ;
        } ;

        this.find_by_status = function (expect_checked) {
            var selector_name = _.find(this._colnames , function (name) {
                return _that._coldef[name].is_selector ;
            }) ;
            return _.filter(this._rows, function (row) {
                var checked = row[selector_name] ;
                return expect_checked ? checked : !checked ;
            }) ;
        } ;
        this.find_checked     = function () { return this.find_by_status(true) ; } ;
        this.find_not_checked = function () { return this.find_by_status(false) ; } ;

        this.check = function (on, predicate) {

            var selector_name = _.find(this._colnames , function (name) {
                return _that._coldef[name].is_selector ;
            }) ;
            _.each(this._rows, function (row) {
                if (predicate(row)) {
                    var toggler = _that._tbody.children('tr[row_id="'+row._row_id+'"]').children('td.check-table-selector').children('div.check-table-toggler') ;
                    if (on) toggler.addClass   ('check-table-toggler-on') ;
                    else    toggler.removeClass('check-table-toggler-on') ;
                    row[selector_name] = on ? true : false ;
                }
            }) ; 
        } ;
        this.check_all = function () {
            this.check(true, function () {
                return true ;
            }) ;
        } ;
        this.uncheck_all = function () {
            this.check(false, function () {
                return true ;
            }) ;
        } ;


        this.find_by_lock_status = function (expect_locked) {
            return _.filter(this._rows, function (row) {
                var locked  = row._locked ;
                return expect_locked ? locked : !locked ;
            }) ;
        } ;
        this.find_locked     = function () { return this.find_by_lock_status(true) ; } ;
        this.find_not_locked = function () { return this.find_by_lock_status(false) ; } ;

        this.lock = function (on, predicate) {

            _.each(this._rows, function (row) {
                if (predicate(row)) {
                    var tr = _that._tbody.children('tr[row_id="'+row._row_id+'"]') ;
                    if (on) tr.addClass   ('check-table-locked') ;
                    else    tr.removeClass('check-table-locked') ;
                    row._locked = on ? true : false ;
                }
            }) ; 
        } ;
        this.lock_all = function () {
            this.lock(true, function () {
                return true ;
            }) ;
        } ;
        this.unlock_all = function () {
            this.lock(false, function () {
                return true ;
            }) ;
        } ;

        // ------------------------
        //   Digest rows (if any)
        // ------------------------

        this.append(rows) ;

        /**
         * @brief Implement the widget rendering protocol as required by
         *        the base class Widget.
         *
         * @returns {undefined}
         */
        this.render = function () {

            if (this._is_rendered) return ;
            this._is_rendered = true ;

            // Lay out the table header

            var html =
'<div class="check-table" > ' +
  '<table>' +
    '<thead>' +
      '<tr>' +
            _.reduce(this._colnames , function (html, name) {
                var col = _that._coldef[name] ;
                if (!col.hidden) html +=
        '<td>'+_.escape(col.text)+'</td>' ;
                return html ;
            }, '') +
      '</tr>' +
    '</thead>' +
    '<tbody></tbody>' +
  '</table>' +
'</div>' ;
            this.container.html(html) ;
            this._tbody = this.container.find('tbody') ;

            // Add the rows (if any provided during the static initialization)

            _.each(this._rows, function (row) {
                _that._display_row(row) ;
            }) ;
        } ;

        this._display_row = function (row, position) {
            var html =
'<tr row_id="'+row._row_id+'" '+(row._locked ? 'class="check-table-locked"' : '')+' data="this row is locked" >' +
                _.reduce (
                    this._colnames ,
                    function (html, name) {
                        var col = _that._coldef[name] ;
                        if (col.is_selector) html +=
  '<td class="check-table-selector" name="'+name+'" align="'+col.align+'" >' +
     '<div style="float:left;"  class="check-table-toggler '+(row[name]   ? ' check-table-toggler-on ' : '')+'" ></div>' +
     '<div style="float:right;" class="check-table-locker" ><span class="ui-icon ui-icon-locked" ></span></div>' +
     '<div style="clear:both;" ></div>' +
  '</td> ' ;
                        else if (col.hidden) ;
                        else                 html +=
  '<td name="'+name+'" align="'+col.align+'" >'+row[name]+'</td> ' ;
                        return html ;
                    }, '') +
'</tr> ' ;
            if (_.isUndefined(position)) {
                this._tbody.append(html) ;
            } else {
                _ASSERT (_.isNumber(position) && position >= 0) ;
                var trs = this._tbody.find('tr') ;
                if (position >= trs.length) {
                    this._tbody.append(html) ;
                } else {
                    $(html).insertBefore(trs[position]) ;
                }
            }
            this._tbody.children('tr[row_id="'+row._row_id+'"]').click(function () {

                    var tr      = $(this) ;
                    var td      = tr.children('td.check-table-selector') ;
                    var toggler = td.children('div.check-table-toggler') ;

                    var row_id  = parseInt(tr.attr('row_id')) ;
                    var name    =          td.attr('name') ;

                    for (var i in _that._rows) {
                        var row = _that._rows[i] ;
                        if ((row._row_id === row_id) && !row._locked) {
                            if (row[name]) toggler.removeClass('check-table-toggler-on') ;
                            else           toggler.addClass   ('check-table-toggler-on') ;
                            row[name] = !row[name] ;
                            break ;
                        }
                    }
                }) ;
        } ;
        this._undisplay_row = function (row) {
            this._tbody.find('tr[row_id="'+row._row_id+'"]').remove() ;
        } ;
    }
    Class.define_class(CheckTable, Widget.Widget, {}, {}) ;

    return CheckTable ;
}) ;