import os
import shutil
import stat
import time
import gc 
from pathlib import Path
import tempfile

class DatasetTransaction:
    def __init__(self, target_dir: str, keep_backup_on_fail: bool = False):
        self.target_dir = Path(target_dir).resolve()
        self.keep_backup = keep_backup_on_fail
        self.temp_dir = None
        self.backup_dir = None

    def __enter__(self):
        if not self.target_dir.exists():
            raise FileNotFoundError(f"Source directory not found: {self.target_dir}")

        self.temp_dir = Path(tempfile.mkdtemp(dir=self.target_dir.parent, prefix=f"tmp_{self.target_dir.name}_"))
        
        print(f"[Transaction] Started. Sandbox: {self.temp_dir.name}")
        print(f"[Transaction] Copying data (this may take time)...")
        
        self._copy_dir(self.target_dir, self.temp_dir)
        return str(self.temp_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect() 

        if exc_type:
            print(f"[Transaction] Error detected: {exc_val}")
            print(f"[Transaction] Rolling back...")
            if self.keep_backup:
                print(f"Sandbox kept for debugging at: {self.temp_dir}")
            else:
                self._force_remove(self.temp_dir)
            return False 

        print(f"[Transaction] Logic successful. Committing...")
        
        try:
            self.backup_dir = self.target_dir.with_name(f"{self.target_dir.name}_backup_{int(time.time())}")

            self._robust_rename(self.target_dir, self.backup_dir)

            try:
                self._robust_rename(self.temp_dir, self.target_dir)
            except Exception as e:
                print("Update failed, attempting to restore backup...")
                self._robust_rename(self.backup_dir, self.target_dir)
                raise e

            print("[Transaction] Cleaning up backup...")
            self._force_remove(self.backup_dir)
            
            print(f"[Transaction] Committed successfully!")
            
        except Exception as e:
            print(f"[Transaction] CRITICAL COMMIT ERROR: {e}")
            raise e

        return True


    def _robust_rename(self, src, dst, retries=5, delay=1.0):
        for i in range(retries):
            try:
                if src.exists():
                    src.rename(dst)
                return
            except PermissionError:
                if i < retries - 1:
                    print(f"File locked. Retrying rename ({i+1}/{retries})...")
                    time.sleep(delay)
                else:
                    raise

    def _copy_dir(self, src, dst):
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    def _force_remove(self, path):
        if not path.exists(): return
        def on_error(func, path, exc_info):
            os.chmod(path, stat.S_IWRITE)
            try:
                func(path)
            except PermissionError:
                print(f"Warning: Could not delete {path} immediately.")
        shutil.rmtree(path, onerror=on_error)