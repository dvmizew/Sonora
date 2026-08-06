import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from sonora.core.constants import USER_AGENT

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})

# Configure connection pooling and retries
retries = Retry(total=3, backoff_factor=0.3, status_forcelist=[500, 502, 503, 504])
adapter = HTTPAdapter(pool_connections=20, pool_maxsize=100, max_retries=retries)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)
