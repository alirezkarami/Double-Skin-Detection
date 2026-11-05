"""Extract a RAR archive using the system libarchive via ctypes (no extra packages)."""
import ctypes
import os
import sys

ARCHIVE_EOF, ARCHIVE_OK, ARCHIVE_WARN = 1, 0, -20

la = ctypes.CDLL("libarchive.so.13")
la.archive_read_new.restype = ctypes.c_void_p
la.archive_write_disk_new.restype = ctypes.c_void_p
la.archive_error_string.restype = ctypes.c_char_p
la.archive_entry_pathname.restype = ctypes.c_char_p
for fn in ("archive_read_support_format_all", "archive_read_support_filter_all",
           "archive_read_close", "archive_read_free", "archive_write_free"):
    getattr(la, fn).argtypes = [ctypes.c_void_p]
la.archive_read_open_filename.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
la.archive_read_next_header.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
la.archive_entry_pathname.argtypes = [ctypes.c_void_p]
la.archive_entry_set_pathname.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
la.archive_write_disk_set_options.argtypes = [ctypes.c_void_p, ctypes.c_int]
la.archive_read_extract2.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
la.archive_error_string.argtypes = [ctypes.c_void_p]

# secure options: no absolute paths, no "..", no symlink traversal
SECURE = 0x0100 | 0x0200 | 0x10000
TIME = 0x0004


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    os.chdir(dst)
    a = la.archive_read_new()
    la.archive_read_support_format_all(a); la.archive_read_support_filter_all(a)
    if la.archive_read_open_filename(a, src.encode(), 1 << 16) != ARCHIVE_OK:
        sys.exit(la.archive_error_string(a))
    w = la.archive_write_disk_new()
    la.archive_write_disk_set_options(w, SECURE | TIME)
    entry = ctypes.c_void_p(); n = 0
    while True:
        r = la.archive_read_next_header(a, ctypes.byref(entry))
        if r == ARCHIVE_EOF:
            break
        if r < ARCHIVE_WARN:
            sys.exit(la.archive_error_string(a))
        r = la.archive_read_extract2(a, entry, w)
        if r < ARCHIVE_WARN:
            print("ERR", la.archive_entry_pathname(entry), la.archive_error_string(a))
        n += 1
    la.archive_read_close(a); la.archive_read_free(a); la.archive_write_free(w)
    print("extracted entries:", n)


if __name__ == "__main__":
    main(os.path.abspath(sys.argv[1]), sys.argv[2])
