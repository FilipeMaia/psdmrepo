/* 
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
function SmartTable (container, hdr, rows, num_hdr_rows, max_hdr_rows) {

    var that = this ;

    this.container = container ;
    this.hdr = hdr || [] ;
    this.rows = rows || [] ;
    this.max_hdr_rows = max_hdr_rows || 10 ;
    this.num_hdr_rows = num_hdr_rows || 1 ;
    this.num_hdr_rows = Math.min(this.max_hdr_rows, 2 + Math.max(Math.floor(hdr.length / 4), this.num_hdr_rows)) ;

    this.load = function (rows) {
        this.rows = rows || [] ;
        this.display() ;
    } ;

    this.display = function () {
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

        switch (typeof this.container) {
            case 'string': this.container = $('#'+this.container) ; break ;
            case 'object': break ;
            default: throw 'SmartTable: wrong type of the container parameter' ;
        }
        this.container.html(html) ;
        this.t_cont = this.container.children('.t-cont') ;
        this.menu   = this.t_cont.find('.menu') ;
        this.header = this.t_cont.find('.header') ;
        this.body   = this.t_cont.find('.body') ;
        this.table  = this.body.find('table') ;
        this.cols   = [] ;
        this.row2td = [] ;
        this.col2td = [] ;

        for (var i in this.rows) {
            this.row2td[i] = this.table.find('td.row_'+i) ;
        }
        for (var i in this.hdr) {
            this.col2td[i] = this.table.find('td.column_'+i) ;
            this.cols[i] = this.header.find('div.column#'+i) ;
            this.cols[i]
                .mouseover(function () {
                    var col = this.id ;
                    that.col2td[col].addClass('selected') ;
                    that.header.find('.column_ext#'+col).addClass('selected') ;
                    if ((that.prev_menu_col !== null) && (that.prev_menu_col !== col)) {
                        if (that.menu_available(col)) that.attach_menu_to(this) ;
                    }
                    var group = Math.floor(col / that.num_hdr_rows) ;
                    that.header.find('.column_group_'+group).addClass('group') ;
                })
                .mouseout(function () {
                    var col = this.id ;
                    that.col2td[col].removeClass('selected') ;
                    that.header.find('.column_ext#'+col).removeClass('selected') ;
                    var group = Math.floor(col / that.num_hdr_rows) ;
                    that.header.find('.column_group_'+group).removeClass('group') ;
                }) ;
            this.col2td[i]
                .mouseover(function () {
                    var row_col = this.id.split(':') ;
                    var row = row_col[0] ,
                        col = row_col[1] ;
                    that.cols[col].addClass('selected') ;
                    that.header.find('.column_ext#'+col).addClass('selected') ;
                    that.row2td[row].addClass('selected') ;
                    that.col2td[col].addClass('selected') ;
                    var group = Math.floor(col / that.num_hdr_rows) ;
                    that.header.find('.column_group_'+group).addClass('group') ;
                    that.table.find('tr#'+row).addClass('selected') ;
                    $(this).addClass('focus') ;
                })
                .mouseout(function () {
                    var row_col = this.id.split(':') ;
                    var row = row_col[0] ,
                        col = row_col[1] ;
                    that.cols[col].removeClass('selected') ;
                    that.header.find('.column_ext#'+col).removeClass('selected') ;
                    that.row2td[row].removeClass('selected') ;
                    that.col2td[col].removeClass('selected') ;
                    var group = Math.floor(col / that.num_hdr_rows) ;
                    that.header.find('.column_group_'+group).removeClass('group') ;
                    that.table.find('tr#'+row).removeClass('selected') ;
                    $(this).removeClass('focus') ;
                }) ;
        }

        this.body.click(function() {
            that.deactivate_menu() ;
        }) ;

        this.header.find('.column').click(function() {
            var col = this.id ;
            if (that.menu_available(col)) that.attach_menu_to(this) ;
        }) ;

        this.menu.find('.item').click(function() {
            var col = that.prev_menu_col ;
            var op = this.id ;
            switch (op) {
                case 'rename' :
                    var elem = that.header.find('.column#'+col).find('.ctrl') ;
                    elem.html(elem.text().substr(0,5)) ;
                    break ;
                case 'front' :
                    var elem = that.header.find('.column#'+col) ;
                    elem.css('z-index', parseInt(elem.css('z-index')) + 1) ;
                    break ;
                case 'back' :
                    var elem = that.header.find('.column#'+col) ;
                    elem.css('z-index', parseInt(elem.css('z-index')) - 1) ;
                    break ;
            }
            that.deactivate_menu() ;
            
        }) ;

        this.t_cont.resize(function () { that.render_header() ; }) ;

        this.render_header() ;
    } ;

    this.prev_menu_col = null ;

    /**
     * Toggle the menu for the specific column and return True if it's on screen.
     *
     * @returns {boolean}
     */
    this.menu_available = function (col) {
        if (this.prev_menu_col === col) {
            this.deactivate_menu() ;
            return false;
        } else {
            this.deactivate_menu() ;
        }
        this.menu.css('display', 'block') ;
        this.prev_menu_col = col ;
        $(document).on('keyup.show_menu', function (e) {
            if (e.keyCode === 27) { that.deactivate_menu() ; }
        }) ;
        return true ;
    } ;
    this.deactivate_menu = function () {
        this.menu.css('display', 'none') ;
        this.prev_menu_col = null ;
        $(document).unbind('keyup.show_menu') ;
    } ;
    this.attach_menu_to = function (e) {
        var offset = $(e).offset() ;

        var ctrl = $(e).find('.ctrl') ;

        var str = ctrl.css('paddingTop') ;
        var paddingTop    = str ? parseInt(str.substr(0,str.length-2)) : 0 ;
        str     = ctrl.css('paddingBottom') ;
        var paddingBottom = str ? parseInt(str.substr(0,str.length-2)) : 0 ;

        that.menu.offset({
           top:  offset.top + parseInt(ctrl.css('height')) + paddingTop + paddingBottom ,
           left: offset.left
        }) ;
        that.menu.css('min-width', ($(e).width() - parseInt(that.menu.css('paddingLeft')) - parseInt(that.menu.css('paddingRight'))) + 'px') ;
    } ;
    this.render_header = function () {
        var column_width = [] ;
        this.table.find('tr:first-child td').each(function () {
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
        this.header.width(this.table.width() - parseInt(this.header.css('borderLeftWidth')) - parseInt(this.header.css('borderRightWidth'))) ;
        var left = 0 ;
        for (var i in this.hdr) {
            var top = 24 * (this.num_hdr_rows - 1) - 24 * (i % this.num_hdr_rows) - 1 ;
            this.cols[i].css('top', top+'px').css('left',left+'px').css('min-width', (column_width[i] -2 )+'px') ;
            //var height = 23 * ((i % this.num_hdr_rows) + 1) + (i % this.num_hdr_rows) ;
            //this.cols[i].height(height) ;
            var column_ext = this.header.find('.column_ext#'+i) ;
            column_ext.css('top', (top + 24 - 1)+'px').css('left',left+'px') ;
            column_ext.width(column_width[i] - 2) ;
            column_ext.height(23 * (i % this.num_hdr_rows) + (i % this.num_hdr_rows) + 1) ;
            left += column_width[i] ;
        }
    } ;
    return this ;
}