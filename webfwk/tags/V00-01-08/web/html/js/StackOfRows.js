define ([
    'webfwk/CSSLoader', 'webfwk/Class', 'webfwk/Widget'] ,

function (
    cssloader, Class, Widget) {

    cssloader.load('../webfwk/css/StackOfRows.css') ;

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
    throw new Widget.WidgetError('RowTitle_Factory: missing or unsupported row title') ;
}

/**
 * The base class representing the body of a row.
 * 
 * Note that this is not the final class as it doesn't implement the 'render' method
 * of the base class 'Widget'.
 * 
 * Also note that properties of this class are set externaly by the StackRowBody_Factory
 * factory function when inserting rows into a stack object, not by derived classes.
 *
 * @returns {StackRowBody}
 * @see {StackRowBody_Factory]
 * @see {Widget]
 */
function StackRowBody () {

    Widget.Widget.call(this) ;
    
    /*
     * The parent stack object
     *
     * @type {StackOfRows}
     */
    this.parent_stack = null ;

    /**
     * A numeric identifier of the row
     *
     * @type {Number}
     */
    this.row_id = null ;
}
Class.define_class(StackRowBody, Widget.Widget, {}, {}) ;

/**
 * The final class to package the body of a stack from 
 * @param {StackOfRows} parent_stack
 * @param {Number} row_id
 * @param {String} text
 * @returns {StackRowBody_FromString}
 * @see {StackRowBody}
 */
function StackRowBody_FromString (parent_stack, row_id, text) {

    StackRowBody.call(this, parent_stack, row_id) ;

    this.text = text ;

    /**
     * Implementing the method of the base class
     *
     * @returns {undefined}
     */
    this.render = function () {
        this.container.html(this.text) ;
    } ;
}
Class.define_class(StackRowBody_FromString, StackRowBody, {}, {}) ;

/**
 * The factory method will return an object coformant to the StackRowBody interface
 *
 * @param {StackOfRows} parent_stack
 * @param {Number} row_id
 * @param {a subclass of StackRowBody, or String or Object} body
 * @returns {Object} an object coformant to the StackRowBody interface
 */
function StackRowBody_Factory (parent_stack, row_id, body) {
    if (body instanceof StackRowBody) {
        body.parent_stack = parent_stack ;
        body.row_id = row_id ;
        return body ;
    }
    switch (typeof(body)) {
        case 'string' : return new StackRowBody_FromString(parent_stack, row_id, body) ;
        case 'object' :
            if (typeof(body.display) === 'function') {
                body.parent_stack = parent_stack ;
                body.row_id = row_id ;
                return body ;
            }
    }
    throw new Widget.WidgetError('StackRowBody_Factory: missing or unsupported row body') ;
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

    StackRowData.call(this) ;

    this.title = title ;
    this.body  = body ;
}
Class.define_class(SimpleStackRowData, StackRowData, {}, {}) ;

/**
 * 
 * @param number id
 * @param object row
 * @param array hdr
 * @param array opt
 * @param function effect_on_insert_required
 * @returns {StackRow}
 */
function StackRowData_Factory (parent, id, row, hdr, opt, effect_on_insert_required) {
    var effect_on_insert = effect_on_insert_required && opt.effect_on_insert ? opt.effect_on_insert : null ;
    if (row instanceof StackRowData) {
        return new StackRow (
            parent ,
            id ,
            row ,
            hdr ,
            opt.expand_propagate ,
            row.color_theme ,
            row.block_common_expand ,
            effect_on_insert ,
            row.block_expand ,
            opt.allow_replicated_headers
        ) ;
    } else if ((typeof row === 'object') && !row.isArray) {
        return new StackRow (
            parent ,
            id ,
            new SimpleStackRowData (
                RowTitle_Factory(row.title, hdr) ,
                StackRowBody_Factory(parent, id, row.body)
            ) ,
            hdr ,
            opt.expand_propagate ,
            row.color_theme ,
            row.block_common_expand ,
            effect_on_insert ,
            row.block_expand ,
            opt.allow_replicated_headers
        ) ;
    }
    throw new Widget.WidgetError('StackRowData_Factory: unsupported type of the row') ;
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
function StackRow (parent, id, data_object, hdr, expand_propagate, color_theme, block_common_expand, effect_on_insert, block_expand, allow_replicated_headers) {

    Widget.Widget.call(this) ;

    this.parent = parent ;
    this.id = id ;
    this.data_object = data_object ;
    this.hdr = hdr ;
    this.expand_propagate = expand_propagate ? (this.data_object.body.expand_or_collapse ? true : false) : false ;
    this.color_theme = color_theme ? color_theme : '' ;
    this.block_common_expand = block_common_expand ? true : false ;
    this.effect_on_insert = effect_on_insert ;
    this.block_expand = block_expand ? true : false ;
    this.allow_replicated_headers = allow_replicated_headers ? true : false ;
}
Class.define_class (StackRow, Widget.Widget, {}, {

/**
 * @function - overloading the function of the base class Widget
 */
render : function () {
    var that = this ;
    var html =
'<div class="stack-row-header">' +
'  <div class="stack-row-column-first">' + (this.block_expand ? '&nbsp;' : '<span class="stack-row-toggler ui-icon ui-icon-triangle-1-e"></span>') + '</div>' +
'  <div class="stack-row-column-last"></div>' +
'</div>' +
'<div class="stack-row-body stack-row-hidden"></div>' ;
    this.container.html(html) ;
    if (this.color_theme) this.container.addClass(this.color_theme) ;

    this.header = this.container.children('.stack-row-header') ;
    this.header.click(function () { that.toggle() ; }) ;
 
    this.toggler = this.header.children('div').children('span.stack-row-toggler') ;
    this.first   = this.header.children('.stack-row-column-first') ;
    this.body    = this.container.children('.stack-row-body') ;

    var html = '' ;
    if (this.hdr) {
        var title_as_function = typeof(this.data_object.title.html) === 'function' ;
        var float_right = false ;   // once triggered, it will shift the remaining columns to the right
        for(var i in this.hdr) {
            var col = this.hdr[i] ;
            if (col.id === '_') {
                html +=
'<div class="stack-row-column" style="min-width:'+col.width+'px;">&nbsp;</div>' ;
            } else if (col.id === '|') {
                html +=
'<div class="stack-row-column-separator">&nbsp;</div>' ;
            } else if (col.id === '>') {
                float_right = true ;
                html +=
'<div class="stack-row-column"></div>' ;
            } else {
                var align = col.align === 'right' ? 'text-align:right;' : '' ;
                html +=
'<div class="stack-row-column'+(float_right ? '-right' : '')+'" id="'+col.id+'" style="min-width:'+col.width+'px; '+align+' '+col.style+'" >'+(title_as_function ? this.data_object.title.html(col.id) : this.data_object.title[col.id])+'</div>' ;
            }
        }
        $(html).insertAfter(this.first) ;
    } else {
        $('<div class="stack-row-column"></div>').insertAfter(this.first) ;
        this.first.next().html(this.data_object.title.html()) ;
    }
    
    // Enfore the header effect if requested and if the effect has been provided
    
    if (this.effect_on_insert) {
        this.effect_on_insert(this.header) ;
    }
} ,

/**
 * @function - update valued of the specified columnd in the table title
 * @param Object title - a map of column identifiers and new values
 */
update_title : function (title, effect_on_update) {
    
    // Replace the title in the local store and render updates

    this.data_object.title = RowTitle_Factory (title, this.hdr) ;
    if (this.hdr) {
        var title_as_function = typeof(this.data_object.title.html) === 'function' ;
        for(var i in this.hdr) {
            var id = this.hdr[i].id ;
            switch (id) {
                case '|' : break ;
                case '>' : break ;
                default:
                    this.header.find('div#'+id).html(title_as_function ? this.data_object.title.html(id) : this.data_object.title[id]) ;
            }
        }
    } else {
        this.first = this.header.children('.stack-row-column-first') ;
        this.first.next().html(this.data_object.title.html()) ;
    }

    // Enfore the header effect if requested and if the effect has been provided
    
    if (effect_on_update)
        effect_on_update(this.header) ;
} ,

/**
 * @function - return the body of the row
 */
get_body : function () {
    return this.data_object.body ;
} ,

/**
 * @function - toggle the row container
 */
toggle : function () {
    var common_expand = false ;
    this.expand_or_collapse(this.toggler.hasClass('ui-icon-triangle-1-e'), common_expand) ;
} ,

/**
 * Expand or collapse the row container
 * 
 * NOTES:
 * - this method will also do lazy rendering of the body first time the expansion operation is called on a row.
 * - the group expand operations may be blocked by the 'block_expand' or 'block_common_expand' parameters of the row
 * - the 'focus_relative_to' parameter only applies to the row expansion operation
 * - the 'focus_relative_to' is ignored if this is a part of the group expand operation
 *
 * @param boolean expand            - expand if 'true', collapse otherwise
 * @param boolean common_expand     - the flag indicating is if this is group operation
 * @param Object  focus_relative_to - (if specified) focus relative to the specified JQuery element
 */
body_is_rendered : false ,
expand_or_collapse : function (expand, common_expand, focus_relative_to) {
    if (this.block_expand) return ;
    if (expand && !(common_expand && this.block_common_expand)) {
        if (!this.body_is_rendered) {
            this.body_is_rendered = true ;
            this.data_object.body.display(this.body) ;
        }
        this.header.addClass('stack-row-header-open') ;
        this.toggler.removeClass('ui-icon-triangle-1-e').addClass('ui-icon-triangle-1-s') ;
        this.body.removeClass('stack-row-hidden').addClass('stack-row-visible') ;
        if (focus_relative_to && !common_expand) {
            var offset_top = focus_relative_to.scrollTop() + this.body.position().top - 24;
            focus_relative_to.animate({scrollTop: offset_top}, 'slow') ;
        }
        if (this.allow_replicated_headers && !common_expand) {
            var replicated_header = this.container.next('.stack-header') ;
            if (!replicated_header.length)
                $(this.parent.stack_header()).insertAfter(this.container) ;
        }
    } else {
        this.header.removeClass('stack-row-header-open') ;
        this.toggler.removeClass('ui-icon-triangle-1-s').addClass('ui-icon-triangle-1-e') ;
        this.body.removeClass('stack-row-visible').addClass('stack-row-hidden') ;
        if (this.allow_replicated_headers) {
            var replicated_header = this.container.next('.stack-header') ;
            if (replicated_header.length)
                replicated_header.remove() ;
        }
    }
    if (this.expand_propagate)
        this.data_object.body.expand_or_collapse(expand) ;
} ,

/**
 * Return 'true' of the row is expanded.
 * 
 * NOTE: this operation won't consider any embedded objects.
 */
is_expanded : function () {
    return this.header.hasClass('stack-row-header-open') ? true : false ;
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

    this.options = {
        expand_buttons          : false ,
        expand_propagate        : false ,
        theme                   : null ,
        hidden_header           : false ,
        effect_on_insert        : null ,
        allow_replicated_headers: false
    } ;
    if (options)
        for(var key in this.options)
            if (key in options)
                this.options[key] = options[key] ;

    this.rows = [] ;
    if (rows)
        for(var i in rows)
            this.add_row(rows[i]) ;

}
Class.define_class (StackOfRows, Widget.Widget, {
    
/******************
 * Static members *
 ******************/

} , {

/***********
 * Methods *
 ***********/


num_rows: function () {
    return this.rows.length ;
} ,

/*
 * The generator of unique identifiers for the rows.
 */
last_id: null ,

next_id: function () {
    if (this.last_id === null) this.last_id = 0 ;
    else this.last_id++ ;
    return this.last_id ;
} ,


add_row : function (row_data) {

    // The identifier is needed by the Stack only in order
    // to keep a track of a container into which a particular
    // row Widget is paced.

    var id = this.next_id() ;

    // The Data Object can be either directly prepared by a user or be
    // "syntechnised" from a simple form of an input.

    this.rows.push (
        StackRowData_Factory (
            this ,
            id ,
            row_data ,
            this.hdr ,
            this.options
        )
    ) ;
    return id ;
} ,

set_rows : function (row_data) {
    this.rows = [] ;
    this.last_id = null ;
    if (row_data)
        for(var i in row_data)
            this.add_row(row_data[i]) ;
} ,

/**
 * This method will return true if there is at least one row in the 'locked' state
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
'</div>' +
    this.stack_header() +
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

stack_header: function () {
    var html = '' ;
    if (this.hdr && !this.options.hidden_header) {
        html +=
'<div class="stack-header">' +
'  <div class="stack-column-first"><span class="stack-header-toggler ui-icon ui-icon-triangle-1-e"></span></div>' ;

        var float_right = false ;   // once triggered, it will shift the remaining columns to the right

        for(var i in this.hdr) {
            var col = this.hdr[i] ;
            if (col.id === '_') {
                html +=
'  <div class="stack-column" style="width:'+col.width+'px;" >&nbsp;</div>' ;
            } else if (col.id === '|') {
                html +=
'  <div class="stack-column-separator">&nbsp;</div>' ;
            } else if (col.id === '>') {
                float_right = true ;
                html +=
'  <div class="stack-column"></div>' ;
            } else {
                var align = col.align === 'right' ? 'text-align:right;' : '' ;
                html +=
'  <div class="stack-column'+(float_right ? '-right' : '')+'" id="'+col.id+'" style="width:'+col.width+'px; '+align+'" >'+col.title+'</div>' ;
            }
        }
        html +=
'  <div class="stack-column-last"></div>' +
'</div>' ;
    }
    return html ;
} ,

assert_initialized: function () {
    if (!this.is_initialized)
        throw new Widget.WidgetError('StackRows.assert_initialized: the widget has not been rendered yet') ;
} ,

get_row_location_by_id: function (id) {
    for(var i in this.rows) {
        var row = this.rows[i] ;
        if (row.id === id) return i ;
    }
    throw new Widget.WidgetError('StackRows.get_row_location_by_id: illegal row id: '+id) ;
} ,

get_row_by_id: function (id) {
    var i = this.get_row_location_by_id(id) ;
    return this.rows[i] ;
} ,

first_row: function () {
    for(var i in this.rows) {
        return this.rows[i] ;
    }
    return null ;
} ,

/* ==========================================================================
 * These operations can be called only after rendering/displaying the widget.
 * Otherwise the method will throw an exception.
 * ==========================================================================
 */

reset: function () {
    this.assert_initialized() ;
    this.body.children('.stack-row').remove() ;
    this.set_rows([]) ;
} ,

/**
 * Expand or collapse all rows
 *
 * @param Boolean expand - a flag indicating a desired operation (true for expanding)
 */
expand_or_collapse : function (expand) {
    this.assert_initialized() ;
    var common_expand = true ;
    for(var i in this.rows) {
        this.rows[i].expand_or_collapse(expand, common_expand) ;
    }
} ,

/**
 * Expand or collapse one row, and optionally focus on to the row after expanding
 *
 * NOTES:
 * - the 'focus_relative_to' parameter only applies to the row expansion operation
 * - the 'focus_relative_to' is ignored if this is a part of the group expand operation
 * 
 * @param Number  id                - a number of the row
 * @param Boolean expand            - a flag indicating a desired operation (true for expanding)
 * @param Object  focus_relative_to - (if specified) focus relative to the specified JQuery element
 */
expand_or_collapse_row : function (id, expand, focus_relative_to) {
    this.assert_initialized() ;
    var row = this.get_row_by_id(id) ;
    var common_expand = false ;
    row.expand_or_collapse(expand, common_expand, focus_relative_to) ;
} ,

/**
 * Return 'true' of the row is expanded. Note that the operation
 * won't consider any embedded objects.
 *
 * @param Number id - a number of the row
 */
is_expanded : function (id) {
    this.assert_initialized() ;
    var row = this.get_row_by_id(id) ;
    return row.is_expanded() ;
} ,

/**
 * Delete the specified row
 * 
 * @param Number id - an identifier of the row
 */
delete_row : function (id) {

    this.assert_initialized() ;

    var i = this.get_row_location_by_id(id) ;

    this.rows.splice(i, 1) ;
    this.body.children('.stack-row#'+id).remove() ;
} ,

/**
 * Update the specified row
 * 
 * @param Number id - an identifier of the row
 * @param Object row_data - the data object to initialize the row
 * @param Boolean effect_on_update_required - teh optional flag indicating if the updated orw shoudl be highlighted
 * @return Number - the unique identifier of the row witin the table
 */
update_row : function (id, row_data, effect_on_update_required) {

    if (!row_data) throw new Widget.WidgetError('StackRows.update_row: illegal row data') ;

    this.assert_initialized() ;

    var i = this.get_row_location_by_id(id) ;
    var old_row = this.rows[i] ;

    var is_expanded = old_row.is_expanded() ;
    var common_expand = false ;

    var effect_on_update_required = effect_on_update_required ? true : false ;
    var new_row_obj = StackRowData_Factory (
        this ,
        id ,
        row_data ,
        this.hdr ,
        this.options ,
        effect_on_update_required
    ) ;
    this.rows[i] = new_row_obj ;

    new_row_obj.display(this.body.children('.stack-row#'+id)) ;
    new_row_obj.expand_or_collapse(is_expanded, common_expand) ;

    return new_row_obj.id ;
} ,

/**
 * Insert a new row in front of the stack
 * 
 * @param Object row_data - the data object to initialize the row
 * @return Number - the unique identifier of the row witin the table
 */
insert_front : function (row_data) {

    if (!row_data) throw new Widget.WidgetError('StackRows.update_row: illegal row data') ;

    this.assert_initialized() ;

    var effect_on_insert_required = true ;
    var new_row_obj = StackRowData_Factory (
        this ,
        this.next_id() ,
        row_data ,
        this.hdr ,
        this.options ,
        effect_on_insert_required
    ) ;

    var html =
'<div class="stack-row" id="'+new_row_obj.id+'"></div>' ;

    var first_row_obj = this.first_row() ;
    if (first_row_obj) {
        this.rows.splice(0, 0, new_row_obj) ;
        $(html).insertBefore(this.body.children('.stack-row#'+first_row_obj.id)) ;
    } else {
        this.rows.push(new_row_obj) ;
        this.body.html(html) ;
    }
    new_row_obj.display(this.body.children('.stack-row#'+new_row_obj.id)) ;

    return new_row_obj.id ;
} ,

/**
 * Append a new row to the stack
 * 
 * @param Object row_data - the data object to initialize the row
 * @return Number - the unique identifier of the row witin the table
 */
append : function (row_data) {

    if (!row_data) throw new Widget.WidgetError('StackRows.append: illegal row data') ;

    this.assert_initialized() ;

    var effect_on_insert_required = false ;     // for now, don't allow any effects when adding rows by
                                                // the end of a stack until ther ewill be a clear
                                                // use case for that.
    var new_row_obj = StackRowData_Factory (
        this ,
        this.next_id() ,
        row_data ,
        this.hdr ,
        this.options ,
        effect_on_insert_required
    ) ;

    var html =
'<div class="stack-row" id="'+new_row_obj.id+'"></div>' ;

    this.rows.push(new_row_obj) ;
    this.body.append(html) ;

    new_row_obj.display(this.body.children('.stack-row#'+new_row_obj.id)) ;
    
    return new_row_obj.id ;
}

}) ;

    return {
        StackRowBody : StackRowBody ,
        StackRowData : StackRowData ,
        StackOfRows : StackOfRows
    } ;
}) ;