"""Read-only EGL device enumeration; creates no rendering context."""
import ctypes as ct
import json
import os
import sys
from pathlib import Path

print('Python:', sys.executable)
for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'LD_LIBRARY_PATH',
            '__EGL_VENDOR_LIBRARY_FILENAMES', 'MESA_D3D12_DEFAULT_ADAPTER_NAME'):
    print('{}={}'.format(key, os.environ.get(key, '<unset>')))
for directory in ('/usr/share/glvnd/egl_vendor.d', '/etc/glvnd/egl_vendor.d'):
    for path in sorted(Path(directory).glob('*.json')):
        print('Vendor {}: {}'.format(path, path.read_text().strip()))
print('WSL devices:', {p: Path(p).exists() for p in ('/dev/dxg', '/dev/nvidia0')})
for directory in ('/usr/lib/wsl/lib', '/usr/lib/x86_64-linux-gnu', str(Path(sys.prefix) / 'lib')):
    libs = sorted(Path(directory).glob('libEGL*'))
    print('EGL libraries in {}: {}'.format(directory, [str(p) for p in libs]))

# Import the simulator so library resolution matches the failing process.
import habitat_sim
egl = ct.CDLL('libEGL.so.1')
egl.eglGetProcAddress.argtypes = [ct.c_char_p]
egl.eglGetProcAddress.restype = ct.c_void_p
egl.eglQueryString.argtypes = [ct.c_void_p, ct.c_int]
egl.eglQueryString.restype = ct.c_char_p
egl.eglGetError.restype = ct.c_uint
print('EGL client extensions:', egl.eglQueryString(None, 0x3055))

def proc(name, result, *arguments):
    address = egl.eglGetProcAddress(name.encode('ascii'))
    if not address:
        raise RuntimeError('Missing EGL function ' + name)
    return ct.CFUNCTYPE(result, *arguments)(address)

query = proc('eglQueryDevicesEXT', ct.c_uint, ct.c_int,
             ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_int))
device_string = proc('eglQueryDeviceStringEXT', ct.c_char_p, ct.c_void_p, ct.c_int)
device_attr = proc('eglQueryDeviceAttribEXT', ct.c_uint, ct.c_void_p, ct.c_int,
                   ct.POINTER(ct.c_ssize_t))
devices = (ct.c_void_p * 16)()
count = ct.c_int()
ok = query(16, devices, ct.byref(count))
print('EGL enumerate:', bool(ok), 'count:', count.value, 'error:', hex(egl.eglGetError()))
for index in range(count.value):
    extensions = device_string(devices[index], 0x3055) or b''
    print('Device', index, 'extensions:', extensions.decode())
    if b'EGL_NV_device_cuda' in extensions:
        cuda_id = ct.c_ssize_t(-1)
        result = device_attr(devices[index], 0x323A, ct.byref(cuda_id))
        print('CUDA mapping:', bool(result), cuda_id.value, hex(egl.eglGetError()))
print('Loaded graphics libraries:')
for line in Path('/proc/self/maps').read_text().splitlines():
    if any(name in line for name in ('libEGL', 'libGLX', 'libGLdispatch', 'libnvidia', 'libd3d12')):
        print(line.split()[-1])
print('HABITAT_EGL_INSPECTION_COMPLETE')
