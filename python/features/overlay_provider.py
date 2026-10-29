"""Route repaired fields to an isolated bundle, all other data to the original."""
from pathlib import Path

from qlib.data.data import LocalFeatureProvider
from qlib.data.storage.file_storage import FileFeatureStorage


class OverlayFeatureProvider(LocalFeatureProvider):
    def __init__(self, overlay, **kwargs):
        super().__init__(**kwargs)
        self.overlay = Path(overlay)

    def feature(self, instrument, field, start_index, end_index, freq):
        name = str(field).lstrip('$').lower()
        path = self.overlay/'features'/str(instrument).lower()/f'{name}.{freq}.bin'
        if path.is_file():
            return FileFeatureStorage(instrument=instrument, field=name, freq=freq,
                                      provider_uri={freq: str(self.overlay)})[start_index:end_index+1]
        return super().feature(instrument, field, start_index, end_index, freq)
