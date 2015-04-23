define ([
    'webfwk/CSSLoader', 'webfwk/Class', 'webfwk/Widget'] ,

function (
    cssloader, Class, Widget) {

    cssloader.load('../webfwk/css/SmartTable.css') ;

    function SmartTable (hdr, rows, num_hdr_rows, max_hdr_rows) {

        var _that = this ;

        // -------------------------------------------
        //   Always call the c-tor of the base class
        // -------------------------------------------

        Widget.Widget.call(this) ;

        // ------------------------------
        //   Data members of the object
        // ------------------------------

        this.hdr = hdr || [] ;
        this.rows = rows || [] ;
        this.max_hdr_rows = max_hdr_rows || 10 ;
        this.num_hdr_rows = num_hdr_rows || 1 ;
        this.num_hdr_rows = Math.min(this.max_hdr_rows, 2 + Math.max(Math.floor(hdr.length / 4), this.num_hdr_rows)) ;

        this._is_rendered = false ; // rendering is done only once

        this._prev_menu_col = null ;
        this._t_cont = null ;
        this._menu   = null ;
        this._header = null ;
        this._body   = null ;
        this._table  = null ;
        this._cols   = [] ;
        this._row2td = [] ;
        this._col2td = [] ;

        /**
         * @brief Implement the widget rendering protocol as required by
         *        the base class Widget.
         *
         * @returns {undefined}
         */
        this.render = function () {

            if (this._is_rendered) return ;
            this._is_rendered = true ;

            this._display() ;
        } ;

        /**
         * @brief Reload the table with new row and redisplay it
         *
         * @param {type} rows
         * @returns {undefined}
         */
        this.load = function (rows) {
            this.rows = rows || [] ;
            this._display() ;
        } ;

        this._display = function () {

            if (!this._is_rendered) return ;

            // Clean up the previous content to make sure eveyting is properly
            // destroyed, including data atatched to DOM, etc. This way we
            // would guarantee that we won't have any memory leaks.
            //
            // NOTE: elementas are destroyed in teh reverse order.

            for (var i in this._cols)   { this._cols  [i].remove() ; }
            for (var i in this._row2td) { this._row2td[i].remove() ; }
            for (var i in this._col2td) { this._col2td[i].remove() ; }
            if (this._table)  this._table.remove() ;
            if (this._body)   this._body.remove() ;
            if (this._header) this._header.remove() ;
            if (this._menu)   this._menu.remove() ;
            if (this._t_cont) this._t_cont.remove() ;

            // Form a new table and set up event handlers

            var html =
'<div class="t-cont">' +
'  <div class="menu">' +
'    <div class="item" id="sort"   >Sort</div>' +
'    <div class="item" id="hide"   >Hide</div>' +
'    <div class="item" id="front"  >Front</div>' +
'    <div class="item" id="back"   >Back</div>' +
'    <div class="item" id="left"   >Move Left</div>' +
'    <div class="item" id="right"  >Move Right</div>' +
'    <div class="item" id="rename" >Rename</div>' +
'    <div class="item" id="rename" >Delete</div>' +
'    <div class="item" id="before" >Insert Column Before</div>' +
'    <div class="item" id="after"  >Insert Column After.</div>' +
'  </div>' +
'  <div class="header" style="height:'+(24*this.num_hdr_rows - 1)+'px;">' +
'    <div class="global_menu" data="table setting and global actions" ></div>' ;
            for (var i in this.hdr) {
                html +=
'    <div class="column column_'+i+' column_group_'+Math.floor(i / this.num_hdr_rows)+'" id="'+i+'" style="z-index:'+(100 - i % this.num_hdr_rows)+';">' +
'      <div class="ctrl">' + this.hdr[i] + '</div>' +
'    </div>' +
'    <div class="column_ext column_ext_'+i+'" id="'+i+'" style="z-index:'+(100 - i % this.num_hdr_rows)+';">&nbsp;</div>' ;
            }
            html +=
'  </div>' +
'  <div class="body">' +
'    <table border="0" cellspacing="0" cellpadding="0" ><tbody>' ;
            for (var i in this.rows) {
                var row = this.rows[i] ;
                html +=
'      <tr id="'+i+'">' ;
                for (var j in row) {
                    var extra_class = j % this.num_hdr_rows ? '' : 'highlight' ;
                    var cell = row[j] ;
                    html +=
'        <td class="row_'+i+' column_'+j+' '+extra_class+'" id="'+i+':'+j+'" align="right" >' +
'          <div data="'+this.hdr[j]+'">' + cell + '</div>' +
'        </td>' ;
                }
                html +=
'      </tr>' ;
            }
            html +=
'    </tbody></table>' +
'  </div>' +
'</div>' ;
            this.container.html(html) ;
            this._t_cont = this.container.children('.t-cont') ;
            this._menu   = this._t_cont.find('.menu') ;
            this._header = this._t_cont.find('.header') ;
            this._body   = this._t_cont.find('.body') ;
            this._table  = this._body.find('table') ;
            this._cols   = [] ;
            this._row2td = [] ;
            this._col2td = [] ;

            for (var i in this.rows) {
                this._row2td[i] = this._table.find('td.row_'+i) ;
            }
            for (var i in this.hdr) {
                this._col2td[i] = this._table.find('td.column_'+i) ;
                this._cols[i] = this._header.find('div.column#'+i) ;
                this._cols[i]
                    .mouseover(function () {
                        var col = this.id ;
                        _that._col2td[col].addClass('selected') ;
                        _that._header.find('.column_ext#'+col).addClass('selected') ;
                        if ((_that._prev_menu_col !== null) && (_that._prev_menu_col !== col)) {
                            if (_that._menu_available(col)) _that._attach_menu_to(this) ;
                        }
                        var group = Math.floor(col / _that.num_hdr_rows) ;
                        _that._header.find('.column_group_'+group).addClass('group') ;
                    })
                    .mouseout(function () {
                        var col = this.id ;
                        _that._col2td[col].removeClass('selected') ;
                        _that._header.find('.column_ext#'+col).removeClass('selected') ;
                        var group = Math.floor(col / _that.num_hdr_rows) ;
                        _that._header.find('.column_group_'+group).removeClass('group') ;
                    }) ;
                this._col2td[i]
                    .mouseover(function () {
                        var row_col = this.id.split(':') ;
                        var row = row_col[0] ,
                            col = row_col[1] ;
                        _that._cols[col].addClass('selected') ;
                        _that._header.find('.column_ext#'+col).addClass('selected') ;
                        _that._row2td[row].addClass('selected') ;
                        _that._col2td[col].addClass('selected') ;
                        var group = Math.floor(col / _that.num_hdr_rows) ;
                        _that._header.find('.column_group_'+group).addClass('group') ;
                        _that._table.find('tr#'+row).addClass('selected') ;
                        $(this).addClass('focus') ;
                    })
                    .mouseout(function () {
                        var row_col = this.id.split(':') ;
                        var row = row_col[0] ,
                            col = row_col[1] ;
                        _that._cols[col].removeClass('selected') ;
                        _that._header.find('.column_ext#'+col).removeClass('selected') ;
                        _that._row2td[row].removeClass('selected') ;
                        _that._col2td[col].removeClass('selected') ;
                        var group = Math.floor(col / _that.num_hdr_rows) ;
                        _that._header.find('.column_group_'+group).removeClass('group') ;
                        _that._table.find('tr#'+row).removeClass('selected') ;
                        $(this).removeClass('focus') ;
                    }) ;
            }

            this._body.click(function() {
                _that._deactivate_menu() ;
            }) ;

            this._header.find('.column').click(function() {
                var col = this.id ;
                if (_that._menu_available(col)) _that._attach_menu_to(this) ;
            }) ;
        

//////////////////////////////////////////////////////////////////////////////
//
//  TODO: Temporarily disable all items in the column menu
//
//            this._menu.find('.item').click(function() {
//                var col = _that._prev_menu_col ;
//                var op = this.id ;
//                switch (op) {
//                    case 'rename' :
//                        var elem = _that._header.find('.column#'+col).find('.ctrl') ;
//                        elem.html(elem.text().substr(0,5)) ;
//                        break ;
//                    case 'front' :
//                        var elem = _that._header.find('.column#'+col) ;
//                        elem.css('z-index', parseInt(elem.css('z-index')) + 1) ;
//                        break ;
//                    case 'back' :
//                        var elem = _that._header.find('.column#'+col) ;
//                        elem.css('z-index', parseInt(elem.css('z-index')) - 1) ;
//                        break ;
//                }
//                _that._deactivate_menu() ;
//
//            }) ;

            this._t_cont.resize(function () { _that._render_header() ; }) ;

            this._render_header() ;
        } ;

        /**
         * Toggle the menu for the specific column and return True if it's on screen.
         *
         * @returns {boolean}
         */
        this._menu_available = function (col) {
            if (this._prev_menu_col === col) {
                this._deactivate_menu() ;
                return false;
            } else {
                this._deactivate_menu() ;
            }
            this._menu.css('display', 'block') ;
            this._prev_menu_col = col ;
            $(document).on('keyup.show_menu', function (e) {
                if (e.keyCode === 27) { _that._deactivate_menu() ; }
            }) ;
            return true ;
        } ;
        this._deactivate_menu = function () {
            this._menu.css('display', 'none') ;
            this._prev_menu_col = null ;
            $(document).unbind('keyup.show_menu') ;
        } ;
        this._attach_menu_to = function (e) {
            var offset = $(e).offset() ;

            var ctrl = $(e).find('.ctrl') ;

            var str = ctrl.css('paddingTop') ;
            var paddingTop    = str ? parseInt(str.substr(0,str.length-2)) : 0 ;
            str     = ctrl.css('paddingBottom') ;
            var paddingBottom = str ? parseInt(str.substr(0,str.length-2)) : 0 ;

            _that._menu.offset({
               top:  offset.top + parseInt(ctrl.css('height')) + paddingTop + paddingBottom ,
               left: offset.left
            }) ;
            _that._menu.css('min-width', ($(e).width() - parseInt(_that._menu.css('paddingLeft')) - parseInt(_that._menu.css('paddingRight'))) + 'px') ;
        } ;
        this._render_header = function () {
            var column_width = [] ;
            this._table.find('tr:first-child td').each(function () {
                var str = $(this).css('borderLeftWidth') ;
                var borderLeft   = str ? parseInt(str.substr(0,str.length-2)) : 0 ;
                str     = $(this).css('paddingLeft') ;
                var paddingLeft  = str ? parseInt(str.substr(0,str.length-2)) : 0 ;
                str     = $(this).css('paddingRight') ;
                var paddingRight = str ? parseInt(str.substr(0,str.length-2)) : 0 ;
                str     = $(this).css('borderRightWidth') ;
                var borderRight  = str ? parseInt(str.substr(0,str.length-2)) : 0 ;
                column_width.push(borderLeft + paddingLeft + $(this).width() + paddingRight + borderRight) ;
            }) ;
            this._header.width(this._table.width() - parseInt(this._header.css('borderLeftWidth')) - parseInt(this._header.css('borderRightWidth'))) ;
            var left = 0 ;
            for (var i in this.hdr) {
                var top = 24 * (this.num_hdr_rows - 1) - 24 * (i % this.num_hdr_rows) - 1 ;
                this._cols[i].css('top', top+'px').css('left',left+'px').css('min-width', (column_width[i] -2 )+'px') ;
                //var height = 23 * ((i % this.num_hdr_rows) + 1) + (i % this.num_hdr_rows) ;
                //this._cols[i].height(height) ;
                var column_ext = this._header.find('.column_ext#'+i) ;
                column_ext.css('top', (top + 24 - 1)+'px').css('left',left+'px') ;
                column_ext.width(column_width[i] - 2) ;
                column_ext.height(23 * (i % this.num_hdr_rows) + (i % this.num_hdr_rows) + 1) ;
                left += column_width[i] ;
            }
        } ;
    }
    Class.define_class(SmartTable, Widget.Widget, {}, {}) ;

    return SmartTable ;

}) ;