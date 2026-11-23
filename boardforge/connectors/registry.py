from boardforge.connectors.ashby import AshbyConnector
from boardforge.connectors.generic import GenericJsonLdConnector
from boardforge.connectors.greenhouse import GreenhouseConnector
from boardforge.connectors.lever import LeverConnector

CONNECTORS = {
    "greenhouse": GreenhouseConnector(),
    "lever": LeverConnector(),
    "ashby": AshbyConnector(),
    "generic": GenericJsonLdConnector(),
}
