"""Exclusive directory creation with fd-relative, no-follow artifact writes."""
import os
from pathlib import Path

from qualification.inputs import safe_open


class Output:
    def __init__(self, path):
        path = Path(os.path.abspath(path))
        parent = safe_open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.mkdir(path.name, mode=0o700, dir_fd=parent)
            self.fd = os.open(path.name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
        finally:
            os.close(parent)
        self.serial = 0

    def write(self, name, data, replace=False):
        if not name or Path(name).name != name:
            raise ValueError('artifact name must be a basename')
        self.serial += 1
        target = f'.receipt-{self.serial}.tmp' if replace else name
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=self.fd)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(data)
            stream.flush()
        if replace:
            os.replace(target, name, src_dir_fd=self.fd, dst_dir_fd=self.fd)

    def close(self):
        os.close(self.fd)
