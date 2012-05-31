/**
 * Utility function for creating base or derived classes.
 * 
 *   function Base(value) {
 *       this.value = value;
 *       Base.num_instances++;
 *   }
 *   define_class( Base, null, {
 *       num_instances: 0
 *   },{
 *       set_value: function(value) { this.value = value; },
 *       get_value: function()      { return this.value;  }}
 *   );
 *   var obj1 = new Base(123);
 *   alert(Base.num_instances);
 *
 *
 *   function Derived(value) {
 *       Base.call(this,value);
 *   }
 *   define_class( Derived, Base, {},{
 *       get_value: function() { return 'Derived: '+this.value;  }}
 *   );
 *   var obj2 = new Derived(1);
 *   alert(Base.num_instances);
 *
 */
function define_class(constructor, base, statics, methods ) {

    if(base) {
        if(Object.create)
            constructor.prototype = Object.create(base.prototype);
        else { 
            function f() {};
            f.prototype = base.prototype;
            constructor.prototype = new f();
        }
    }
    constructor.prototype.constructor = constructor;

    if(statics)
        for(var s in statics)
            constructor[s] = statics[s];

    if(methods)
        for(var m in methods)
            constructor.prototype[m] = methods[m];

    return constructor;
}

/**
 * The Table class:
 * 
 *   Table(id,coldef,data,text_when_empty)
 * 
 * The class is designed to simplify creating dynamic tables
 * at a location specified by HTML container 'id'. Table configuration
 * is expected in parameter 'coldef' which is a JSON object. The JSON
 * object is a dictionary of column definitions:
 * 
 *   [ <col-def-0>, <col-def-1> .. <col-def-N> ]
 * 
 * Each column definition is a dictionary which allows to define either
 * a simple column:
 * 
 *   { name: <col-name>, type: <type-descriptor> }
 * 
 * or a composite column:
 * 
 *   { name: <col-name>, coldef: [ <col-def-0>, <col-def-1> .. <col-def-M> ] }
 *
 * Definitions for the bottom level columns may also provide additional parameters
 * to override the defaults:
 * 
 * 1. Data elements type:
 * 
 *      type: <type-descriptor>
 *
 *    Where, type descriptior can be either one of the predefined types:
 *
 *      Table.Types.Number_URL
 *      Table.Types.Number
 *      Table.Types.Text      <- the default type if none is used
 *      Table.Types.Text_URL
 *
 *    Where type Number_URL corresponds to the following data objects:
 *
 *      var value {
 *        number: 123,            // some number
 *        url:  "https://..."     // the corresponding URL
 *        ..
 *      };
 *
 *    Where the last type (Text_URL) corresponds to the following data objects:
 *
 *      var value {
 *        text: "123",           // some text
 *        url:  "https://..."    // the corresponding URL
 *        ..
 *      };
 *
 *    It's also possible to pass the type descriptor as a user defined dictionary
 *    of optional functions describing how to interpret data elements for
 *    a custom type and (optionally) what to do after sorting.
 *    For example, of a data element is an object:
 *    
 *      var value = {
 *        value: 123,
 *        url: 'https://....'
 *      };
 *
 *    then the type descriptor should look like:
 *
 *      type: {
 *        to_string:      function(a)   { return '<a href="'+a.url+'"; target="_blank";'>'+a.value+'</a>'; },
 *        compare_values: function(a,b) { return a.value - b.value; },
 *      }
 *
 *    Note, that all above mentioned members of the type descriptor
 *    are optional. If 'to_string' is not provided then the default method 'toString()'
 *    will be assumed. If 'compare_values' is not provided then value object will be
 *    used, which may result in unpredictable sort order of rows.
 *
 *    And finaly, there is an alternative method of creating a custom
 *    cell class. This would provide maximum flexibility when defining
 *    customazable and tunable column types. In theory one can imagine
 *    customising column types on the fly by communicating with those
 *    custom objects and triggering table redisplay.
 *
 *      function MyCellType() { TableCellType.call(this); }
 *      define_class( MyCellType, TableCellType, {}, {
 *        to_string     : function(a)   { return '<button class="my_button" name="'+a+'">'+a+'</button>'; },
 *        compare_values: function(a,b) { return this.compare_strings(a,b); },
 *        after_sort    : function()    { $('.my_button').button(); }}
 *      );
 *    The last parameter is optional. It will be triggered each time rows
 *    have been sorted. It allows to dynamically customize cells of
 *    thsi type.
 *
 *    Finally, instantiate an object and pass it as the column parameter:
 *
 *      type: new MyCellType()
 *    
 * 2. Sorting flag:
 *
 *      sorted: {true|false}
 *    
 *    Where the default is 'true'.
 *
 * Other parameters:
 *
 *   'data'            - data array to be preloaded when creating the table
 *   'text_when_empty' - HTML text to show when no data are loaded
 */

function TableCellType() {}
define_class( TableCellType, null, {}, {
    to_string:       function(a)   { return ''+a; },
    compare_numbers: function(a,b) { return a - b; },
    compare_strings: function(a,b) {
        var a_ = ''+a;
        var b_ = ''+b;
        return ( a_ < b_ ) ? -1 : (( b_ < a_ ) ? 1 : 0 ); },
    compare_values: function(a,b) { return this.compare_strings(a,b); },
    after_sort    : function() {}}
);

function TableCellType_Number() { TableCellType.call(this); }
define_class( TableCellType_Number, TableCellType, {}, {
    compare_values: function(a,b) { return this.compare_numbers(a,b); }}
);

function TableCellType_NumberURL() { TableCellType.call(this); }
define_class( TableCellType_NumberURL, TableCellType, {}, {
    compare_values: function(a,b) { return this.compare_numbers(a.number,b.number); },
    to_string     : function(a)   { return '<a class="table_link" href="'+a.url+'"; target="_blank";>'+a.number+'</a>'; }}
);

function TableCellType_Text() { TableCellType.call(this); }
define_class( TableCellType_Text, TableCellType, {}, {});

function TableCellType_TextURL() { TableCellType.call(this); }
define_class( TableCellType_TextURL, TableCellType, {}, {
    to_string     : function(a)   { return '<a class="table_link" href="'+a.url+'"; target="_blank";>'+a.text+'</a>'; },
    compare_values: function(a,b) { return this.compare_strings(a.text,b.text); }}
);


function Table(id,coldef,data,text_when_empty) {

    /** 
     * Constructor
     */

    // Mandatory parameters of the table

    this.id     = id;                               // container address where to render the table
    this.coldef = jQuery.extend(true,[],coldef);    // columns definition: make a deep local copy

    // Optional parameters

    this.data            = data ? data : [];
    this.text_when_empty = text_when_empty ? text_when_empty : Table.Status.Empty;


    // Sort configuration

    this.sorted = {
        column:  0,     // the number of a column by which rows are sorted
        forward: true   // sort direction
    };

    this.header = {
        size: {cols: 0, rows: 0},
        types: [],
        sorted: []
    };
    this.header.size  = this.header_size(this.coldef);

    var bottom_columns = this.column_types(
        this.coldef,
        this.header.types,
        this.header.sorted,
        0
    );
    this.header.types  = bottom_columns.types;
    this.header.sorted = bottom_columns.sorted;
}
define_class( Table, null, {

/******************
 * Static members *
 ******************/

    Types: {
        Number:     new TableCellType_Number(),
        Number_URL: new TableCellType_NumberURL(),
        Text:       new TableCellType_Text(),
        Text_URL:   new TableCellType_TextURL()},

    Status: {
        Empty  : '&lt;&nbsp;'+'empty'+'&nbsp;&gt;',
        Loading: '&nbsp;'+'loading...'+'&nbsp;',
        error  : function(msg) { return '<span style="color:red;">&lt;&nbsp;'+msg+'&nbsp;&gt;</span>'; }},

    sort_func: function(type,column) {
        this.compare = function(a,b) {
            return type.compare_values(a[column], b[column] );
        }},

    sort_sign_classes_if: function(condition,forward) {
        return condition ? ['ui-icon',(forward ? 'ui-icon-triangle-1-s' : 'ui-icon-triangle-1-n')] : []; }

},{

/***********
 * Methods *
 ***********/

    cols: function() {
        return this.header.size.cols;
    },

    sort_data: function() {
        var column = this.sorted.column;
        if( !this.header.sorted[column] ) return;
        var bound_sort_func = new Table.sort_func(this.header.types[column], column);
        this.data.sort( bound_sort_func.compare );
        if( !this.sorted.forward ) this.data.reverse();
    },

    load: function(data) {
        this.data = data ? data : [];
        this.display();
    },

    display: function() {

        /**
         * Render the table within a container provided as a parameter
         * of the table object. Each header cell is located at a level
         * which varies from 0 up to the total number of the full header's
         * rows minus 1.
         * 
         * NOTE: that because multi-level rows in HTML tables are produced
         * by walking an upper layer first and gradually procinging to lower-level
         * rows then we only drow header cells at the requested level.
         */

        var that = this;
        var html = '<table><tbody>';

        // Draw header

        for( var level2drows=0; level2drows < this.header.size.rows; ++level2drows ) {
            html += '<tr>';
            for( var i in this.coldef ) {
                var col = this.coldef[i];
                html += this.display_header(0,level2drows,col);
            }
            html += '</tr>';
        }

        // Draw data rows (if available)

        if( this.data.length ) {
            this.sort_data();
            for( var i in this.data ) {
                html += '<tr>';
                var row = this.data[i];
                for( var j=0; j < row.length; ++j ) {
                    var classes = ' table_cell table_bottom';
                    if( j == 0 ) classes += ' table_cell_left';
                    if( j == row.length - 1 ) classes += ' table_cell_right';
                    html += '<td class="'+classes+'">'+this.header.types[j].to_string(row[j])+'</td>';
                }
                html += '</tr>';
            }
        } else {
            if( this.text_when_empty )
                html +=
                    '<tr>'+
                    '<td class="table_cell" colspan='+this.cols()+' rowspan=1 >'+this.text_when_empty+'</td>'+
                    '</tr>';
        }
        html += '</tbody></table>';

        $('#'+this.id).html(html);
        $('#'+this.id).find('.table_row_sorter').click(function() {
            var column = parseInt($(this).find('.table_sort_sign').attr('name'));
            that.sorted.forward = !that.sorted.forward;
            that.sorted.column  = column;
            that.sort_data();
            that.display();
        });
        for( var i in this.header.types ) {
            this.header.types[i].after_sort(); 
        }
    },

    erase: function(text_when_empty) {
        if( text_when_empty ) this.text_when_empty = text_when_empty;
        this.load([]);
    },

    display_header: function(level,level2drows,col) {
        var html = '';
        var rowspan = this.header.size.rows - level;
        var colspan = 1;
        if( col.coldef ) {
            var child = this.header_size(col.coldef);
            rowspan -= child.rows;  // minus rows for children
            colspan = child.cols;   // columns for children
        }

        // Drowing is only done if we're at the right level

        if( level == level2drows ) {
            var align = colspan > 1 ? 'align="center"' : '';
            var classes = ' table_hdr';
            var sort_sign = '';
            if( rowspan + level == this.header.size.rows ) {
                if( this.header.sorted[col.number] ) {
                    classes += ' table_active_hdr table_row_sorter';
                    var sort_sign_classes = Table.sort_sign_classes_if(this.sorted.column == col.number, this.sorted.forward);
                    sort_sign = '<span class="table_sort_sign';
                    for( var i in sort_sign_classes) sort_sign += ' '+sort_sign_classes[i];
                    sort_sign += '" name="'+col.number+'">'+'</span>';
                }
            }
            html += '<td class="'+classes+'" rowspan='+rowspan+' colspan='+colspan+' '+align+' >'+
                '<div style="float:left;">'+col.name+'</div><div style="float:left;">'+sort_sign+'</div><div style="clear:both;"></div></td>';
        }

        // And to optimize things we stop walking the header when teh level drops
        // below the level where we're supposed to drow things.

        if(( level2drows > level ) && col.coldef ) {
            for( var i in col.coldef ) {
                var child_col = col.coldef[i];
                html += this.display_header(level+1,level2drows,child_col);
            }
        }
        return html;
    },

    header_size: function(coldef) {

        /**
         * Traverse colum definition and return the maximum limits
         * for the table header, including:
         * 
         *   rows: the number of rows needed to represent the full header
         *   cols: the total number of low-level columns for the data
         */

        var rows2return = 0;
        var cols2return = 0;
        for( var i in coldef ) {
            var col  = coldef[i];
            var rows = 1;
            var cols = 1;
            if( col.coldef ) {
                var child = this.header_size(col.coldef);
                rows += child.rows;
                cols  = child.cols;
            }
            if( rows > rows2return ) rows2return = rows;
            cols2return += cols;
        }
        return {rows: rows2return, cols: cols2return};
    },

    column_types: function(coldef, types, sorted, next_column_number) {

        /**
         * Traverse colum definition and return types for the bottom-most
         * header cells.
         */

        for( var i in coldef ) {
            var col = coldef[i];
            if( col.coldef ) {
                var child          = this.column_types(col.coldef, types, sorted, next_column_number);
                types              = child.types;
                sorted             = child.sorted;
                next_column_number = child.next_column_number;
            } else {
                if(col.type) {
                    if($.isPlainObject(col.type)) {
                        var type = function() { TableCellType.call(this); };
                        define_class(type, TableCellType, {}, col.type);
                        types.push(new type());
                    } else {
                        types.push(col.type);
                    }
                } else {
                    types.push(Table.Types.Text);
                }
                sorted.push(col.sorted !== undefined ? col.sorted : true);
                col.number = next_column_number++;
            }
        }
        return {
            types:              types,
            sorted:             sorted,
            next_column_number: next_column_number
        };
    }}
);

function Attributes_HTML(attr) {
    var html = '';
    if(attr) {
        if(attr.classes)  html += ' class="'+attr.classes+'"';
        if(attr.name)     html += ' name="'+attr.name+'"';
        if(attr.title)    html += ' title="'+attr.title+'"';
        if(attr.disabled) html += ' disabled="disabled"';
        if(attr.checked)  html += ' checked="checked"';
        if(attr.onclick)  html += ' onclick="'+attr.onclick+'"';
    }
    return html;
}
function Checkbox_HTML(attr) {
    var html = '<input type="checkbox"'+Attributes_HTML(attr)+'/>';
    return html;
}
function Button_HTML(name,attr) {
    var html = '<button '+Attributes_HTML(attr)+'>'+name+'</button>';
    return html;
}