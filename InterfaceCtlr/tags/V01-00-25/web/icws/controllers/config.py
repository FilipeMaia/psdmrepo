#--------------------------------------------------------------------------
# File and Version Information:
#  $Id$
#
# Description:
#  Module config...
#
#------------------------------------------------------------------------

"""Pylons controller for the Interface Controller request-level resource.

This software was developed for the SIT project.  If you use all or 
part of it, please give an appropriate acknowledgment.

@see RelatedModule

@version $Id$

@author Andrei Salnikov
"""


#------------------------------
#  Module's version from CVS --
#------------------------------
__version__ = "$Revision$"
# $Source$

#--------------------------------
#  Imports of standard modules --
#--------------------------------
import sys

#---------------------------------
#  Imports of base class module --
#---------------------------------
from icws.lib.base import BaseController
import formencode
import simplejson
from urllib import quote, unquote

#-----------------------------
# Imports for other modules --
#-----------------------------
from pylons.decorators import jsonify
from pylons.controllers.util import abort
from icws.lib.base import *
from icws.model.icdb_model import IcdbModel

#----------------------------------
# Local non-exported definitions --
#----------------------------------

def _sectId(name):
    if name: return name
    return "__empty__"

def _sectName(id):
    if id == "__empty__": return ""
    return id

def _quote_attr(val):
    if not val: return '""'
    if '"' not in val: return '"'+val+'"'
    if "'" not in val: return "'"+val+"'"
    return '"'+val.replace('"', '&quot;')+'"'

_xsl_base     = '/xsl/default.xsl'
_xsl_default  = '<?xml-stylesheet type="text/xsl" title="Default Style" href="%s"?>'
def _xsl():
    return _xsl_default % h.url_for(_xsl_base)


class _CreateNewParamForm(formencode.Schema):
    allow_extra_fields = True
    filter_extra_fields = True
    if_key_missing = None
    section = formencode.validators.String(not_empty=False)
    param = formencode.validators.String(not_empty=True)
    value = formencode.validators.String(not_empty=False)
    type = formencode.validators.OneOf(['Integer','Float','String','Date/Time'])
    description = formencode.validators.String(not_empty=True)
    instrument = formencode.validators.String(not_empty=False, if_empty=None)
    experiment = formencode.validators.String(not_empty=False, if_empty=None)

class _UpdateParamForm(formencode.Schema):
    allow_extra_fields = True
    filter_extra_fields = True
    if_key_missing = None
    value = formencode.validators.String(not_empty=False)
    instrument = formencode.validators.String(not_empty=False, if_empty=None)
    experiment = formencode.validators.String(not_empty=False, if_empty=None)

class _DeleteParamForm(formencode.Schema):
    allow_extra_fields = True
    filter_extra_fields = True
    if_key_missing = None
    instrument = formencode.validators.String(not_empty=False, if_empty=None)
    experiment = formencode.validators.String(not_empty=False, if_empty=None)


def _renderParmList(params, section_name, renderer):

    # render
    if renderer == 'xml':
        response.content_type = 'application/xml'
        res = ['<?xml version="1.0" encoding="UTF-8"?>']
        res += ['<config-section id="%s" section="%s">' % (_sectId(section_name), section_name)]
        for parm in params:
            parmstr = ["<config-param"]
            for k, v in parm.items():
                parmstr.append('%s=%s' % (k, _quote_attr(v)))
            res.append(' '.join(parmstr) +'/>')
        res += ['</config-section>']
        return '\n'.join(res)
    else:
        response.content_type = 'application/json'
        return simplejson.dumps(params)

#------------------------
# Exported definitions --
#------------------------

#---------------------
#  Class definition --
#---------------------
class ConfigController(BaseController):

    #-------------------
    #  Public methods --
    #-------------------

    @h.catch_all
    def index(self, renderer="json"):
        """
        Get the list of sections.
        """

        def sdict(sect):
            section_id = _sectId(sect)
            url = h.url_for(action="section", section_id=section_id)
            return dict(name=sect, id=section_id, url=url)

        h.checkAccess('', '', 'config')

        model = IcdbModel()

        sections = model.get_config_sections()        
        sections = [sdict(sect) for sect in sorted(sections)]

        if renderer == 'xml':
            response.content_type = 'application/xml'
            res = ['<?xml version="1.0" encoding="UTF-8"?>\n', '<config-sections>\n']
            res += ['  <config-section id="%(id)s" name="%(name)s" url="%(url)s"/>\n' % sec for sec in sections]
            res += ['</config-sections>\n']
            return res
        else:
            response.content_type = 'application/json'
            return simplejson.dumps(sections)

    @h.catch_all
    def section(self, section_id, renderer="json"):
        """
        Get the list of parameters in one section.
        """

        h.checkAccess('', '', 'config')

        model = IcdbModel()

        # map section id to name
        section_name = _sectName(section_id)

        # get all params, filter only those from given section        
        params = model.get_config(section_name)
        if not params:
            abort(404, "Section name not found in configuration table: "+section_name)

        # remove section name from dict, add param ID which is the same as param name
        for parm in params: 
            del parm['section']
            parm['section_id'] = section_id
            parm['id'] = parm.get('param')
            parm['url'] = h.url_for("param_url", section_id=section_id, param_id=parm['id'], renderer='json', qualified=True)
        params.sort(key=lambda o: o.get('param'))

        # render
        return _renderParmList(params, section_name, renderer)


    @h.catch_all
    def parameter(self, section_id, param_id, renderer="json"):
        """
        Get the list of parameters in one section.
        """

        h.checkAccess('', '', 'config')

        model = IcdbModel()

        # map section id to name
        section_name = _sectName(section_id)

        # get all params, filter only those from given section        
        params = model.get_config(section_name)
        params = [p for p in params if p.get('param') == param_id]
        if not params:
            abort(404, "Section or parameter name not found in configuration table: [%s].%s" % (section_name, param_id))

        # remove section name from dict, add param ID which is the same as param name
        for parm in params: 
            del parm['section']
            parm['id'] = parm.get('param')
            parm['section_id'] = section_id
            parm['url'] = h.url_for("param_url", section_id=section_id, param_id=parm['id'], renderer=renderer, qualified=True)
        params.sort(key=lambda o: o.get('param'))

        # render
        return _renderParmList(params, section_name, renderer)


    @h.catch_all
    def create(self, renderer='json'):
        """
        Get the list of parameters in one section.
        """

        # validate parameters
        schema = _CreateNewParamForm()
        try:
            form_result = schema.to_python(dict(request.params))
        except formencode.Invalid, error:
            abort(400, str(error))

        section_name = form_result['section']
        param = form_result['param']
        value = form_result['value']
        type = form_result['type']
        description = form_result['description']
        instrument = form_result['instrument']
        experiment = form_result['experiment']

        h.checkAccess('', '', 'config')

        # map section id to name
        section_id = _sectId(section_name)

        try:
            model = IcdbModel()
            model.create_config(section_name, param, value, type, description, instrument, experiment)
        except Exception, ex:
            abort(400, str(ex))

        # return value
        params = [dict(
            id=param,
            section_id=section_id,
            param=param,
            value=value,
            type=type,
            description=description,
            instrument=instrument or "",
            experiment=experiment or "",
            url = h.url_for("param_url", section_id=section_id, param_id=param, renderer=renderer, qualified=True)
            )]

        # render
        return _renderParmList(params, section_name, renderer)



    @h.catch_all
    def update(self, section_id, param_id, renderer='json'):
        """
        Get the list of parameters in one section.
        """

        # validate parameters
        schema = _UpdateParamForm()
        try:
            form_result = schema.to_python(dict(request.params))
        except formencode.Invalid, error:
            abort(400, str(error))

        value = form_result['value']
        instrument = form_result['instrument']
        experiment = form_result['experiment']

        h.checkAccess('', '', 'config')

        # map section id to name
        section_name = _sectName(section_id)

        try:
            model = IcdbModel()
            model.update_config(section_name, param_id, value, instrument, experiment)
        except Exception, ex:
            abort(400, str(ex))

        # return value
        params = [dict(
            id=param_id,
            section_id=section_id,
            param=param_id,
            value=value,
            instrument=instrument or "",
            experiment=experiment or "",
            url = h.url_for("param_url", section_id=section_id, param_id=param_id, renderer=renderer, qualified=True)
            )]

        # render
        return _renderParmList(params, section_name, renderer)


    @h.catch_all
    def delete(self, section_id, param_id, renderer='json'):
        """
        Get the list of parameters in one section.
        """

        # validate parameters
        schema = _DeleteParamForm()
        try:
            form_result = schema.to_python(dict(request.params))
        except formencode.Invalid, error:
            abort(400, str(error))

        instrument = form_result['instrument']
        experiment = form_result['experiment']

        h.checkAccess('', '', 'config')

        # map section id to name
        section_name = _sectName(section_id)

        try:
            model = IcdbModel()
            model.delete_config(section_name, param_id, instrument, experiment)
        except Exception, ex:
            abort(400, str(ex))

        # return value
        params = [dict(
            id=param_id,
            section_id=section_id,
            param=param_id,
            instrument=instrument or "",
            experiment=experiment or "",
            )]

        # render
        return _renderParmList(params, section_name, renderer)



    @h.catch_all
    def show_full(self, renderer="json"):

        # even to show 
        h.checkAccess('', '', 'config')

        model = IcdbModel()

        data = model.get_config()
        sections = {}
        for d in data:
            section_id = _sectId(d['section'])
            d['section_id'] = section_id
            d['url'] = h.url_for("param_url", section_id=section_id, param_id=d['param'], renderer='json', qualified=True)
            sections.setdefault(d['section'], []).append(d)
            del d['section']

        if renderer == 'xml':
            response.content_type = 'application/xml'
            res = ['<?xml version="1.0" encoding="UTF-8"?>\n', _xsl(), '<config-sections>\n']
            for sec in sorted(sections.keys()):
                res.append('<config-section id="%s" name="%s">\n' % (_sectId(sec), sec))
                sections[sec].sort(key=lambda o: o.get('param'))
                for opt in sections[sec]:
                    optstr = ["<config-param"]
                    for k, v in opt.items():
                        optstr.append('%s="%s"' % (quote(k), v or ""))
                    optstr = ' '.join(optstr) +'/>\n'
                    res.append(optstr)
                res.append('</config-section>\n')
            res += ['</config-sections>\n']
            return res
        else:
            response.content_type = 'application/json'
            return simplejson.dumps(sections)
