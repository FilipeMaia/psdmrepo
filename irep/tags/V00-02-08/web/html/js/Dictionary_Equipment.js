define ([
    'webfwk/CSSLoader' ,
    'webfwk/Class', 'webfwk/FwkApplication', 'webfwk/Fwk', 'webfwk/SimpleTable'] ,

function (
    cssloader ,
    Class, FwkApplication, Fwk, SimpleTable) {

    cssloader.load('../irep/css/Dictionary_Equipment.css') ;

    /**
     * The application for browsing and managing a dictionary of models and manufacturers
     *
     * @returns {Dictionary_Equipment}
     */
    function Dictionary_Equipment (app_config) {

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

        // Automatically refresh the page at specified interval only

        this._update_ival_sec = 120 ;
        this._prev_update_sec = 0 ;

        this.on_update = function () {
            if (this.active) {
                var now_sec = Fwk.now().sec ;
                if (now_sec - this._prev_update_sec > this._update_ival_sec) {
                    this._prev_update_sec = now_sec ;
                    this._init() ;
                    this._load() ;
                }
            }
        } ;

        // -----------------------------
        // Parameters of the application
        // -----------------------------

        this._app_config = app_config ;

        // ----------------
        // Public interface
        // ----------------

        /**
         * Preload the dictionary w/o displaying it
         *
         * @returns {undefined}
         */
        this.init = function () {
            this._preload() ;
        } ;

        this.manufacturers = function () {
            return this._manufacturer ;
        } ;

        /**
         * Return a dictionary with a manufacturer and a model if available.
         * 
         * NOTE: the function must be called after loading the dictionary.
         *
         * @param {number} id
         * @returns {object}
         */
        this.find_model_by_id = function (id) {
            if (this._manufacturer) {
                for (var i in this._manufacturer) {
                    var manufacturer = this._manufacturer[i] ;
                    for (var j in manufacturer.model) {
                        var model = manufacturer.model[j] ;
                        if (id == model.id) {
                            return {
                                manufacturer: manufacturer ,
                                model:        model
                            } ;
                            break ;
                        }
                    }
                }
            }
            return null ;
        } ;

        /**
         * Return a dictionary with a manufacturer and a model if available.
         * 
         * NOTE: the function must be called after loading the dictionary.
         *
         * @param {string} manuf_name
         * @param {string} model_name
         * @returns {string}
         */
        this.find_model = function (manuf_name, model_name) {
            if (this._manufacturer) {
                for (var i in this._manufacturer) {
                    var manufacturer = this._manufacturer[i] ;
                    if (manufacturer.name == manuf_name) {
                        for (var j in manufacturer.model) {
                            var model = manufacturer.model[j] ;
                            if (model.name == model_name) {
                                return {
                                    manufacturer: manufacturer ,
                                    model:        model
                                } ;
                                break ;
                            }
                        }
                    }
                }
            }
            return null ;
        } ;

        // --------------------
        // Own data and methods
        // --------------------

        this._is_initialized = false ;

        this._manufacturer = null ;

        this._can_manage = function () { return this._app_config.current_user.has_dict_priv ; } ;

        this._wa = function (html) {
            if (!this._wa_elem) {
                var html = html ||
'<div id="dictionary-equipment" >' +

  '<div style="float:right;" >' +
    '<button name="update" class="control-button" title="update from the database" >UPDATE</button>' +
  '</div>' +
  '<div style="clear:both;" ></div>' +

  '<div id="updated-info" >' +
    '<div class="info" id="info"    style="float:left;"  >&nbsp;</div>' +
    '<div class="info" id="updated" style="float:right;" >&nbsp;</div>' +
    '<div style="clear:both;" ></div>' +
  '</div>' +

  '<div id="tables" >' +

    '<div class="table-cont" style="float:left;" >' +
      '<div style="float:left; "><input type="text" size="12" name="manufacturer" title="fill in new manufacturer name, then press RETURN to save" /></div>' +
      '<div style="float:left; padding-top:4px; color:maroon;" >  &larr; add new manufacturer here</div>' +
      '<div style="clear:both; "></div>' +
      '<div id="table-manufacturer" class="table" ></div>' +
    '</div>' +

    '<div class="table-cont" style="float:left;" >' +
      '<div style="float:left; "><input type="text" size="12" name="model" title="fill in new model name, then press RETURN to save" /></div>' +
      '<div style="float:left; padding-top:4px; color:maroon;" >  &larr; add new model here</div>' +
      '<div style="clear:both; "></div>' +
      '<div id="table-model" class="table" ></div>' +
    '</div>' +

    '<div style="clear:both;" ></div>' +

  '</div>' +

'</div>' ;
                this.container.html(html) ;
                this._wa_elem = this.container.children('#dictionary-equipment') ;
            }
            return this._wa_elem ;
        } ;
        this._set_info = function (html) {
            if (!this._info_elem) this._info_elem = this._wa().find('#info') ;
            this._info_elem.html(html) ;
        } ;
        this._set_updated = function (html) {
            if (!this._updated_elem) this._updated_elem = this._wa().find('#updated') ;
            this._updated_elem.html(html) ;
        } ;
        this._table_manufacturer = function () {
            if (!this._table_manufacturer_obj) {
                this._table_manufacturer_elem = this._wa().find('#table-manufacturer') ;
                var hdr =  [] ;
                if (this._can_manage()) hdr.push (
                    {   name: 'DELETE', hideable: true, sorted: false ,
                        type: {
                            after_sort: function () {
                                _that._table_manufacturer_elem.find('.manufacturer-delete').button().click (function () {
                                    var id = this.name ;
                                    _that._manufacturer_delete(id) ;
                                }) ;
                            }
                        }
                    }) ;
                hdr.push (
                    {   name: 'manufacturer', selectable: true ,
                        type: {
                            select_action : function (manufacturer_name) {
                                _that._model_display() ;
                            }
                        }
                    } ,
                    {   name: 'created', hideable: true } ,
                    {   name: 'by user', hideable: true } ,
                    {   name: 'description', hideable: true, sorted: false ,
                        type: {
                            after_sort: function() {
                                for (var i in _that._manufacturer) {
                                    var manufacturer = _that._manufacturer[i] ;
                                    _that._table_manufacturer_elem.find('#manufacturer-description-'+manufacturer.id).val(manufacturer.description) ;
                                }
                                _that._table_manufacturer_elem.find('.manufacturer-description-save').button().click (function() {
                                    var id = this.name ;
                                    _that._save_manufacturer_description(id, _that._table_manufacturer_elem.find('#manufacturer-description-'+id).val()) ;
                                });
                            }
                        }
                    } ,
                    {   name: 'in use',  hideable: true, sorted: false ,
                        type: {
                            after_sort: function () {
                                _that._table_manufacturer_elem.find('.manufacturer-search').button().click (function () {
                                    var id = this.name ;
                                    global_search_equipment_by_manufacturer(id) ;
                                }) ;
                            }
                        }
                    }
                ) ;
                var rows = [] ;
                this._table_manufacturer_obj = new SimpleTable.constructor (
                    this._table_manufacturer_elem ,
                    hdr ,
                    rows ,
                    {selected_col: this._can_manage() ? 1 : 0} ,
                    Fwk.config_handler('dict', 'table_manufacturer')
                ) ;
                this._table_manufacturer_obj.display() ;
            }
            return this._table_manufacturer_obj ;
        } ;
        this._table_model = function () {
            if (!this._table_model_obj) {
                this._table_model_elem = this._wa().find('#table-model') ;
                var hdr =  [] ;
                if (this._can_manage()) hdr.push (
                    {   name: 'DELETE', hideable: true, sorted: false ,
                        type: {
                            after_sort: function () {
                                _that._table_model_elem.find('.model-delete').button().click (function () {
                                    var id = this.name ;
                                    _that._model_delete(id) ;
                                }) ;
                                _that._table_model_elem.find('.model-search').button().click (function () {
                                    var id = this.name ;
                                    global_search_equipment_by_model(id) ;
                                }) ;
                            }
                        }
                    }) ;
                hdr.push (
                    {   name: 'model',       style: 'font-weight:bold;' } ,
                    {   name: 'created',     hideable: true } ,
                    {   name: 'by user',     hideable: true } ,
                    {   name: 'description', hideable: true, sorted: false ,
                        type: {
                            after_sort: function() {
                                var manufacturer_name = _that._table_manufacturer().selected_object() ;
                                for (var i in _that._manufacturer) {
                                    var manufacturer = _that._manufacturer[i] ;
                                    if (manufacturer.name === manufacturer_name) {
                                        for (var j in manufacturer.model) {
                                            var model = manufacturer.model[j] ;
                                            _that._table_model_elem.find('#model-description-'+model.id).val(model.description) ;
                                        }
                                    }
                                }
                                _that._table_model_elem.find('.model-description-save').button().click (function() {
                                    var id = this.name;
                                    _that._save_model_description(id, _that._table_model_elem.find('#model-description-'+id).val());
                                });
                            }
                        }
                    } ,
                    {   name: 'image', hideable: true, sorted: false ,
                        type: {
                            after_sort: function () {
                                _that._table_model_elem.find('.model-image-delete').button().click (function () {
                                    var attachment_id = this.name ;
                                    _that._model_image_delete(attachment_id) ;
                                }) ;
                                _that._table_model_elem.find('.model-image-upload').button().click (function () {
                                    var model_id = this.name ;
                                    _that._model_image_upload(model_id) ;
                                }) ;
                            }
                        }
                    } ,
                    {   name: 'in use',  hideable: true, sorted: false }
                ) ;
                var rows = [] ;
                this._table_model_obj = new SimpleTable.constructor (
                    this._table_model_elem ,
                    hdr ,
                    rows ,
                    {} ,
                    Fwk.config_handler('dict', 'table_model')
                ) ;
                this._table_model_obj.display() ;
            }
            return this._table_model_obj ;
        } ;
        this._init = function () {

            if (this._is_initialized) return ;
            this._is_initialized = true ;

            this._wa().find('button[name="update"]').button().click(function () {
                _that._load() ;
            }) ;
            this._wa().find('input[name="manufacturer"]').keyup(function (e) {
                var name = $(this).val() ;
                if (name === '') { return ; }
                if (e.keyCode === 13) { _that._manufacturer_create(name) ; $(this).val('') ; return ; }
            }) ;
            this._wa().find('input[name="model"]').keyup(function (e) {
                var name = $(this).val() ;
                if (name === '') { return ; }
                if (e.keyCode === 13) { _that._model_create(name) ; $(this).val('') ; return ; }
            }) ;

            this._load() ;
        } ;
        this._load = function () {
            if (!this._is_initialized) return ;
            this._manufacturer_action (
                'Loading...' ,
                '../irep/ws/manufacturer_get.php' ,
                {}
            ) ;
        } ;
        this._preload = function () {
            var dont_display = true ;
            this._manufacturer_action (
                'Loading...' ,
                '../irep/ws/manufacturer_get.php' ,
                {} ,
                dont_display
            ) ;
        } ;
        this._manufacturer_display = function () {

            var rows = [] ;
            for (var i in this._manufacturer) {

                var manufacturer = this._manufacturer[i] ;
                var row = [] ;

                if (this._can_manage()) row.push (
                    SimpleTable.html.Button ('X', {
                        name:    manufacturer.id,
                        classes: 'control-button control-button-small control-button-important manufacturer-delete',
                        title:   'delete this manufacturer from the list' })) ;

                row.push (
                    manufacturer.name ,
                    manufacturer.created_time ,
                    manufacturer.created_uid) ;

                if (this._can_manage()) row.push (
                    '<div style="float:left;">' +
                        SimpleTable.html.TextArea ({
                            id:      'manufacturer-description-'+manufacturer.id ,
                            classes: 'description' } ,
                            4 ,
                            36) +
                    '</div>' +
                    '<div style="float:left; margin-left:5px;">' +
                        SimpleTable.html.Button ('save', {
                            name:    manufacturer.id ,
                            classes: 'manufacturer-description-save' ,
                            title:   'edit description for the manufacturer' }) +
                    '</div>' +
                    '<div style="clear:both;">') ;
                else row.push (
                    '<div style="width:256px; overflow:auto;"><pre>'+manufacturer.description+'</pre></div>') ;

                row.push (
                    SimpleTable.html.Button ('search', {
                        name:    manufacturer.id,
                        classes: 'manufacturer-search',
                        title:   'search for all equipment of this manufacturer' })
                ) ;
                rows.push(row) ;
            }
            this._table_manufacturer().load(rows, true) ;   // keep the previous selection

            this._model_display() ;

            if (this._can_manage()) {
                var input = this._wa().find('input[name="model"]') ;
                if (this._table_manufacturer().selected_object() === null) input.attr('disabled', 'disabled') ;
                else                                                       input.removeAttr('disabled') ;
            }
        } ;
        this._model_display = function () {

            var rows = [] ;

            var manufacturer_name = this._table_manufacturer().selected_object() ;
            if (manufacturer_name !== null) {
                for (var i in this._manufacturer) {
                    var manufacturer = this._manufacturer[i] ;
                    if (manufacturer.name === manufacturer_name) {
                        for (var j in manufacturer.model) {
                            var model = manufacturer.model[j] ;
                            var row = [] ;
                            if (this._can_manage()) row.push (
                                SimpleTable.html.Button ('X', {
                                    name:    model.id,
                                    classes: 'control-button control-button-small control-button-important model-delete',
                                    title:   'delete this model from the list' })) ;
                            var images_html =
                                this._can_manage()
                                ?   model.default_attachment.is_available
                                    ?   '<div style="float:left;">' +
                                          SimpleTable.html.Button('delete', {
                                              name:    model.default_attachment.id,
                                              classes: 'model-image-delete',
                                              title:   'delete this image' }) +
                                        '</div>' +
                                        '<div style="float:left; margin-left:10px;">' +
                                        '  <a class="link" href="../irep/model_attachments/'+model.default_attachment.id+'/file" target="_blank" title="click on the image to open/download a full size image in a separate tab"><img src="../irep/model_attachments/preview/'+model.default_attachment.id+'" /></a>' +
                                        '</div>' +
                                        '<div style="clear:both;"></div>'
                                    :   '<div style="float:left;">' +
                                          SimpleTable.html.Button('upload', {
                                              name:    model.id,
                                              classes: 'model-image-upload visible',
                                              title:   'upload image' }) +
                                        '</div>'+
                                        '<div class="hidden" style="float:left; margin-left:5px;" id="model-image-upload-'+model.id+'">' +
                                        '  <form enctype="multipart/form-data" action="../irep/ws/model_image_upload.php" method="post">' +
                                        '    <input type="hidden" name="model_id" value="'+model.id+'" />' +
                                        '    <input type="hidden" name="MAX_FILE_SIZE" value="25000000" />' +
                                        '    <input type="file" name="file2attach" onchange="Fwk.get_application(\''+this.application_name+'\',\''+this.context1_name+'\')._model_image_submit('+model.id+')" />' +
                                        '    <input type="hidden" name="file2attach" value="" />' +
                                        '  </form>' +
                                        '</div>' +
                                        '<div style="clear:both;"></div>'
                                :   model.default_attachment.is_available
                                    ?   '<a class="link" href="../irep/model_attachments/'+model.default_attachment.id+'/file" target="_blank" title="click on the image to open/download a full size image in a separate tab"><img src="../irep/model_attachments/preview/'+model.default_attachment.id+'" /></a>'
                                    :   '' ;
                            row.push (
                                model.name ,
                                model.created_time ,
                                model.created_uid ,

                                this._can_manage() ?
                                    '<div style="float:left;">' +
                                        SimpleTable.html.TextArea ({
                                            id:      'model-description-'+model.id ,
                                            classes: 'description' } ,
                                            4 ,
                                            36) +
                                    '</div>' +
                                    '<div style="float:left; margin-left:5px;">' +
                                        SimpleTable.html.Button ('save', {
                                            name:    model.id ,
                                            classes: 'model-description-save' ,
                                            title:   'edit description for the model' }) +
                                    '</div>' +
                                    '<div style="clear:both;">' :
                                    '<div style="width:256px; overflow:auto;"><pre>'+model.description+'</pre></div>' ,

                                images_html ,

                                SimpleTable.html.Button ('search', {
                                    name:    model.id,
                                    classes: 'model-search',
                                    title:   'search for all equipment of this manufacturer and model' })
                            ) ;
                            rows.push(row) ;
                        }
                        break ;
                    }
                }
            }
            this._table_model().load(rows) ;
        } ;
        this._manufacturer_create = function (name) {
            this._manufacturer_action (
                'Creating...' ,
                '../irep/ws/manufacturer_new.php' ,
                {name: name}
            ) ;
        } ;
        this._manufacturer_delete = function (id) {
            Fwk.ask_yes_no (
                'Confirm Manufacturer Deletion' ,
                'Are you sure you want to delete this manufacturer from the Dictionary?' ,
                function () {
                    _that._manufacturer_action (
                        'Deleting...' ,
                        '../irep/ws/manufacturer_delete.php' ,
                        {id: id}
                    ) ;
                }
            ) ;
        } ;
        this._save_manufacturer_description = function (id,description) {
            this._manufacturer_action_POST (
                'Saving...' ,
                '../irep/ws/manufacturer_update.php' ,
                {id: id, description: description}
            ) ;
        } ;
        this._model_create = function (name) {
            this._model_action (
                'Creating...' ,
                '../irep/ws/model_new.php' ,
                {manufacturer_name: this._table_manufacturer().selected_object(), model_name: name}
            ) ;
        } ;
        this._model_delete = function (id) {
            Fwk.ask_yes_no (
                'Confirm Model Deletion' ,
                'Are you sure you want to delete this model from the Dictionary?' ,
                function () {
                    _that._model_action (
                        'Deleting...' ,
                        '../irep/ws/model_delete.php' ,
                        {id: id}
                    ) ;
                }
            ) ;
        } ;
        this._save_model_description = function (id,description) {
            this._model_action_POST (
                'Saving...' ,
                '../irep/ws/model_update.php' ,
                {id: id, description: description}
            ) ;
         } ;
        this._model_image_upload = function (model_id) {
            this._table_model().get_container().find('#model-image-upload-'+model_id)                 .removeClass('hidden') .addClass('visible')
            this._table_model().get_container().find('button.model-image-upload[name="'+model_id+'"]').removeClass('visible').addClass('hidden') ;
        } ;
        this._model_image_submit = function (model_id) {
            this._table_model().get_container().find('#model-image-upload-'+model_id).find('form').ajaxSubmit({
                success: function(data) {
                    if (data.status != 'success') {
                        Fwk.report_error(data.message) ;
                    } else {
                        _that._manufacturer = data.manufacturer ;
                    }
                    _that._model_display() ;
                } ,
                error: function() {
                    Fwk.report_error('failed to contact the server in order to upload the image') ;
                } ,
                dataType: 'json'
            }) ;
        } ;
        this._model_image_delete = function (attachment_id) {
            this._model_action (
                'Deleting...' ,
                '../irep/ws/model_image_delete.php' ,
                {id: attachment_id}
            ) ;
        } ;
        this._manufacturer_action = function (name, url, params, dont_display) {
            if (dont_display) {
                Fwk.web_service_GET (url, params, function (data) {
                    _that._manufacturer = data.manufacturer ;
                }) ;
            } else {
                this._set_updated(name) ;
                Fwk.web_service_GET (url, params, function (data) {
                    _that._manufacturer = data.manufacturer ;
                    _that._manufacturer_display() ;
                    _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
                }) ;
            }
        } ;
        this._manufacturer_action_POST = function (name, url, params) {
            this._set_updated(name) ;
            Fwk.web_service_POST(url, params, function (data) {
                _that._manufacturer = data.manufacturer ;
                _that._manufacturer_display() ;
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }) ;
        } ;
        this._model_action = function (name, url, params) {
            this._set_updated(name) ;
            Fwk.web_service_GET(url, params, function (data) {
                _that._manufacturer = data.manufacturer ;
                _that._model_display() ;
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }) ;
        } ;
        this._model_action_POST = function (name, url, params) {
            this._set_updated(name) ;
            Fwk.web_service_POST(url, params, function (data) {
                _that._manufacturer = data.manufacturer ;
                _that._model_display() ;
                _that._set_updated('[ Last update on: <b>'+data.updated+'</b> ]') ;
            }) ;
        } ;
    } ;
    Class.define_class (Dictionary_Equipment, FwkApplication, {}, {}) ;
    
    return Dictionary_Equipment ;
}) ;

