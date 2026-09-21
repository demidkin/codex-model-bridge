"""Kernel-owned process leases; process exit releases locks without PID guessing."""
import fcntl
import hashlib
import os
from pathlib import Path


class Lease:
    def __init__(self, directory, name):
        root = Path(directory) / 'locks'
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if root.is_symlink() or root.stat().st_uid != os.getuid():
            raise ValueError('Unsafe lock directory ownership')
        root.chmod(0o700)
        self.path = root / (hashlib.sha256(name.encode()).hexdigest() + '.lock')
        self.file = None

    def acquire(self):
        if self.file is not None:
            return True
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            return False
        except BaseException:
            os.close(fd)
            raise
        self.file = fd
        return True

    def close(self):
        if self.file is not None:
            os.close(self.file)
            self.file = None

    def __enter__(self):
        if not self.acquire():
            raise ValueError('This task is active in another adapter process')
        return self

    def __exit__(self, *args):
        self.close()


def is_owned(directory, name):
    lease = Lease(directory, name)
    if not lease.acquire():
        return True
    lease.close()
    return False
