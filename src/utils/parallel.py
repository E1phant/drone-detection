from os import getenv

from joblib import Parallel
from tqdm import tqdm


class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        # This global environment variable can be used to disable tqdm progress bars across the codebase.
        # for non-interactive terminals, printing the progress bars prints many new lines
        # instead of updating the same line.
        tqdm_disable_globally = getenv("TQDM_DISABLE", "0").lower() in (
            "true",
            "1",
            "t",
        )
        self._use_tqdm = use_tqdm and not tqdm_disable_globally
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        try:
            with tqdm(
                disable=not self._use_tqdm, total=self._total
            ) as self._pbar:
                return Parallel.__call__(self, *args, **kwargs)
        finally:
            # Ensure the bar is closed cleanly
            try:
                self._pbar.close()
            except Exception:
                pass

    def print_progress(self):
        if not self._use_tqdm:
            return super().print_progress()
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()
