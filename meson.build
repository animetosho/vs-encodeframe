project('EncodeFrame', 'cpp',
  default_options: ['buildtype=release', 'warning_level=2', 'b_lto=true', 'b_ndebug=if-release', 'cpp_std=c++14', 'cpp_rtti=false', 'cpp_eh=none'],
  license: 'CC0',
  meson_version: '>=0.51.0',
  version: '2'
)

static = get_option('static')
isa = get_option('isa')

vapoursynth_dep = dependency('vapoursynth', version: '>=55').partial_dependency(compile_args: true, includes: true)

deps = [
  vapoursynth_dep,
  dependency('libturbojpeg', version: '>=1.2.0', static: static),
]

install_dir = vapoursynth_dep.get_variable(pkgconfig: 'libdir') / 'vapoursynth'

sources = [
  'encodeframe.cpp',
  'fpnge/fpnge.cc'
]

if host_machine.cpu_family().startswith('x86')
  if isa == 'sse4'
    add_project_arguments('-msse4.1', language: 'cpp')
  endif
  if isa == 'avx'
    add_project_arguments(['-mavx','-mpclmul'], language: 'cpp')
  endif
  if isa == 'avx2'
    add_project_arguments(['-mavx2','-mpclmul'], language: 'cpp')
  endif
endif

if static
  add_project_link_arguments('-static', language: 'cpp')
endif

shared_module('encodeframe', sources,
  dependencies: deps,
  install: true,
  install_dir: install_dir,
  gnu_symbol_visibility: 'hidden'
)
