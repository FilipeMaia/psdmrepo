/*
 * Two wrapper classes and a factory for producing an appropriate
 * RowTitle object from the specified input.
 */
function RowTitle_FromString (text) {
    this.text = text ;
    this.html = function () { return this.text ; } ;
}
function RowTitle_FromObject (obj) {
    this.obj = obj ;
    this.html = function (id) { return this.obj[id] ; } ;
}
function RowTitle_Factory (title, hdr) {
    if (hdr) {
        switch (typeof(title)) {
            case 'object' : return new RowTitle_FromObject(title) ;
        }
    } else {
        switch (typeof(title)) {
            case 'string' : return new RowTitle_FromString(title) ;
        }
    }
    throw new WidgetError('RowTitle_Factory: missing or unsupported row title') ;
}

/*
 * One wrapper classe and a factory for producing an appropriate
 * RowBody object from the specified input.
 */
function RowBody_FromString (text) {
    this.text = text ;
    this.display = function (container) { container.html(this.text) ; } ;
}
function RowBody_Factory (body) {
    switch (typeof(body)) {
        case 'string' : return new RowBody_FromString(body) ;
        case 'object' : if (typeof(body.display) === 'function') return body ;
    }
    throw new WidgetError('RowBody_Factory: missing or unsupported row body') ;
}

/**
 * The base class for user defined data objects
 *
 * @returns {StackRowData}
 */
function StackRowData () {
    this.title = null ;
    this.body  = null ;
    this.is_locked = function () { return false ; } ;
}

/**
 * The class packaging the title and the body into a single object
 * 
 * NOTE: No run time synchronization is assumed between the components of
 * the row.
 *
 * @param object title
 * @param object body
 * @returns {SimpleStackRowData}
 */
function SimpleStackRowData (title, body) {
    this.title = title ;
    this.body  = body ;
}
define_class(SimpleStackRowData, StackRowData, {}, {}) ;

/**
 * 
 * @param number id
 * @param object row
 * @param array hdr
 * @param boolean expand_propagate
 * @returns {StackRow}
 */
function StackRowData_Factory (id, row, hdr, expand_propagate) {
    if (row instanceof StackRowData) {
        return new StackRow (
            id ,
            row ,
            hdr ,
            expand_propagate ,
            row.color_theme
        ) ;
    } else if ((typeof row === 'object') && !row.isArray) {
        return new StackRow (
            id ,
            new SimpleStackRowData (
                RowTitle_Factory(row.title, hdr) ,
                RowBody_Factory(row.body)
            ) ,
            hdr ,
            expand_propagate ,
            row.color_theme
        ) ;
    }
    throw new WidgetError('StackRowData_Factory: unsupported type of the row') ;
}

/**
 * The class representing rows in the stack
 *
 * A row is described by a data object, which composes two components
 * each responsing to method 'display(container)' similarily to the Widget.
 * Here is an expected interface of the data object:
 * 
 *   { title: {
 *       display: function(container) {...}} ,
 *     body: {
 *       display: function(container) {...}}
 *   }
 *
 * Other aspects of a particular implementation of this data object
 * are irrelevant in the present context. In principle, any of its components
 * may be real Widget objects.
 *
 * @param Number id - is an identifier of an object in the owner's context
 * @param Array data_object - the data object descriibing the title and the body of the row
 * @returns {StackRow}
 */
function StackRow (id, data_object, hdr, expand_propagate, color_theme) {
    this.id = id ;
    this.data_object = data_object ;
    this.hdr = hdr ;
    this.expand_propagate = expand_propagate ? (this.data_object.body.expand_or_collapse ? true : false) : false ;
    this.color_theme = color_theme ? color_theme : '' ;
}
define_class (StackRow, Widget, {}, {

/**
 * @function - overloading the function of the base class Widget
 */
render : function () {
    var that = this ;
    var html =
'<div class="stack-row-header">' +
'  <div style="float:left;"><span class="stack-row-toggler ui-icon ui-icon-triangle-1-e"></span></div>' +
'  <div style="float:left;" class="stack-row-title"></div>' +
'  <div style="clear:both;"></div>' +
'</div>' +
'<div class="stack-row-body stack-row-hidden"></div>' ;
    this.container.html(html) ;
    if (this.color_theme) this.container.addClass(this.color_theme) ;

    this.header = this.container.children('.stack-row-header') ;
    this.header.click(function () { that.toggle() ; }) ;
 
    this.toggler = this.header.children('div').children('span.stack-row-toggler') ;
    this.title   = this.header.children('.stack-row-title') ;
    this.body    = this.container.children('.stack-row-body') ;

    if (this.hdr) {
        var title_as_function = typeof(this.data_object.title.html) === 'function' ;
        var html = '' ;
        var float_right = false ;   // once triggered, it will shift the remaining columns to the right
        for(var i in this.hdr) {
            var col = this.hdr[i] ;
            if (col.id === '|') {
                html +=
'  <div class="stack-row-column-separator">&nbsp</div>' ;
            } else if (col.id === '>') {
                float_right = true ;
                html +=
'  <div class="stack-row-column"></div>' ;
            } else {
                html +=
'  <div class="stack-row-column'+(float_right ? '-right' : '')+'" id="'+col.id+'" style="width:'+col.width+'px;" >'+(title_as_function ? this.data_object.title.html(col.id) : this.data_object.title[col.id])+'</div>' ;
            }
        }
        html +=
'  <div class="stack-row-column-last"></div>';
        this.title.html(html) ;
    } else {
        this.title.html(this.data_object.title.html()) ;
    }
    this.data_object.body.display(this.body) ;
} ,

/**
 * @function - toggle the row container
 */
toggle : function () {
    this.expand_or_collapse(this.toggler.hasClass('ui-icon-triangle-1-e')) ;
} ,

/**
 * Expand or collapse the row container
 */
expand_or_collapse : function (expand) {
    if (expand) {
        this.header.addClass('stack-row-header-open') ;
        this.toggler.removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s') ;
        this.body.removeClass('stack-row-hidden').addClass('stack-row-visible') ;
    } else {
        this.header.removeClass('stack-row-header-open') ;
        this.toggler.removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e') ;
        this.body.removeClass('stack-row-visible').addClass('stack-row-hidden') ;
    }
    if (this.expand_propagate)
        this.data_object.body.expand_or_collapse(expand) ;
} ,

/**
 * Return 'true' if the row is locked from any updates either by the user data
 * or because its body is open.
 */
is_locked : function () {
    return this.data_object.is_locked() || this.toggler.hasClass('ui-icon-triangle-1-s') ;
}

}) ;

/**
 * A class representing a stackable table of rows
 *
 * @param Array - an array of the header columns (optional)
 * @param Array - an array of rows (optional)
 * @param Array opt - a dictionary of options
 * @returns {StackOfRows}
 */
function StackOfRows (hdr, rows, options) {

    this.hdr = hdr ? hdr : null ;

    this.rows = [] ;
    if (rows)
        for(var i in rows)
            this.add_row(rows[i]) ;

    this.options = {
        expand_buttons : false ,
        expand_propagate : false ,
        theme : null ,
        hidden_header : false
    } ;
    if (options)
        for(var key in this.options)
            if (key in options)
                this.options[key] = options[key] ;

}
define_class (StackOfRows, Widget, {
    
/******************
 * Static members *
 ******************/

} , {

/***********
 * Methods *
 ***********/

add_row : function (row) {

    // The identifier is needed by the Stack only in order
    // to keep a track of a container into which a particular
    // row Widget is paced.

    var id = this.rows.length ;

    // The Data Object can be either directly prepared by a user or be
    // "syntechnised" from a simple form of an input.

    this.rows.push (
        StackRowData_Factory (
            id ,
            row ,
            this.hdr ,
            this.options.expand_propagate
        )
    ) ;
} ,

set_rows : function (rows) {
    this.rows = [] ;
    if (rows)
        for(var i in rows)
            this.add_row(rows[i]) ;
} ,

/**
 * This method will return true if ther eis at least one row in the 'locked' state
 * which is meant to prevent the table from being deleted/updated, etc.
 */
is_locked : function () {
    for (var i in this.rows)
        if (this.rows[i].is_locked()) return true ;
    return false ;
} ,

is_initialized : false ,

initialize : function () {
    if (this.is_initialized) return ;
    this.is_initialized = true ;
} ,

/**
 * Overloaded function of the base class Widget
 */
render : function () {

    this.initialize() ;
    var that = this ;

    // Deploy the current theme at the container

    if (this.options.theme) this.container.addClass(this.options.theme) ;

    // Render the stack layout first

    var html =
'<div class="stack-controls">' ;
    if (this.options.expand_buttons) html +=
'  <button name="expand"   title="expand all rows"   >expand</button>' +
'  <button name="collapse" title="collapse all rows" >collapse</button>' ;
    html +=
'</div>' ;
    if (this.hdr && !this.options.hidden_header) {
        html +=
'<div class="stack-header">' +
'  <div class="stack-column-first"><span class="stack-header-toggler ui-icon ui-icon-triangle-1-e"></span></div>' ;

        var float_right = false ;   // once triggered, it will shift the remaining columns to the right

        for(var i in this.hdr) {
            var col = this.hdr[i] ;
            if (col.id === '|') {
                html +=
'  <div class="stack-column-separator">&nbsp</div>' ;
            } else if (col.id === '>') {
                float_right = true ;
                html +=
'  <div class="stack-column"></div>' ;
            } else {
                html +=
'  <div class="stack-column'+(float_right ? '-right' : '')+'" id="'+col.id+'" style="width:'+col.width+'px;" >'+col.title+'</div>' ;
            }
        }
        html +=
'  <div class="stack-column-last"></div>' +
'</div>' ;
    }
    html +=
'<div class="stack-body" >' ;
    for(var i in this.rows) html +=
'  <div class="stack-row" id="'+this.rows[i].id+'" >' +
'  </div>' ;
    html +=
'</div>' ;

    this.container.html(html) ;

    // Render each row in its container created in the Stack body

    this.body = this.container.children('.stack-body') ;

    for(var i in this.rows) {
        var row = this.rows[i] ;
        row.display(this.body.children('.stack-row#'+row.id)) ;
    }

    // Register event handlers

    this.controls = this.container.children('.stack-controls') ;

    if (this.options.expand_buttons) {
        this.controls.children('button[name="expand"]'  ).button().click(function () {
            that.expand_or_collapse(true) ;
        }) ;
        this.controls.children('button[name="collapse"]').button().click(function () {
            that.expand_or_collapse(false) ;
        }) ;
    }

    this.header = this.container.children('.stack-header') ;
    this.header.click(function () {
        var toggler = $(this).children('.stack-column-first').children('span.stack-header-toggler') ;
        if (toggler.hasClass('ui-icon-triangle-1-e')) {
            toggler.removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s') ;
            that.expand_or_collapse(true) ;
        } else {
            toggler.removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e') ;
            that.expand_or_collapse(false) ;
        }
    }) ;
} ,

/**
 * Expand or collapse all rows
 *
 * @param Boolean expand - a flag indicating a desired operation (true for expanding)
 */
expand_or_collapse : function (expand) {
    for(var i in this.rows)
        this.rows[i].expand_or_collapse(expand) ;
}

}) ;