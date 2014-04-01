"""
Tool supporting Qt4.
"""
import os
from SCons.Builder import Builder
from SCons.Action import Action

from SConsTools.trace import *
from SConsTools.scons_functions import *

# Qt4 directory location relative to SIT_ROOT
qt_dir = "sw/external/qt"
qt_ver = "4.8.5"

def _qtdir(env):
    for arch in env['SIT_ARCH'], env['SIT_ARCH_BASE'], env['SIT_ARCH_BASE']+'-opt':
        p = os.path.join(env['SIT_ROOT'], qt_dir, qt_ver, arch)
        trace ( "checking QTDIR="+p, "qt4", 2 )
        if os.path.isdir(p): return p

mocAction = Action("$QT4_MOCCOM")

def create_builder(env):
    try:
        moc = env['BUILDERS']['Moc']
    except KeyError:
        moc = Builder( action = mocAction,
                  emitter = {},
                  suffix = ".cpp",
                  single_source = 1)
        env['BUILDERS']['Moc'] = moc

    return moc

def generate(env):
    
    qtdir = _qtdir(env)
    if not qtdir: fail("Cannot determine QTDIR")

    # extend CPPPATH
    p = os.path.join(qtdir, 'include')
    incdirs = [p, pjoin(p, 'QtCore'), pjoin(p, 'QtGui')]
    
    # set env
    env["QTDIR"] = qtdir
    env["QT4_PREFIX"] = os.path.join(env['SIT_ROOT'], qt_dir, qt_ver)
    env["QT4_VER"] = qt_ver
    env["QT4_MOC"] = "$QTDIR/bin/moc"
    env["QT4_MOCCOM"] = "$QT4_MOC $QT4_MOCFLAGS -o $TARGET $SOURCE"
    env["QT4_MOCFLAGS"] = ""
    env["QT4_LIBS"] = ["QtGui", "QtCore"]
    env["QT4_LIBDIR"] = os.path.join(qtdir, "lib")
    env["QT4_INCDIRS"] = incdirs
    env.Append(CPPPATH = incdirs)

    create_builder(env)

    trace ( "Initialized qt4 tool", "qt4", 2 )

def exists(env):
    return _qtdir(env) is not None
