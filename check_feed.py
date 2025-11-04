# Fix para usar datos IEX en lugar de SIP
import sys
sys.path.insert(0, 'src')

from data.ingest import DataIngestionManager
import inspect

# Ver si hay alguna configuración de feed
print(inspect.getsource(DataIngestionManager.get_realtime_data))
