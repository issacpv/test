"""NeuroMorpho.Org REST client: filtered morphology search and SWC retrieval.

NeuroMorpho.Org (Ascoli et al. 2007, J Neurosci; Akram et al. 2018, Sci Data)
exposes a HAL-style REST API. The two calls this project needs are

* ``GET /api/neuron/select?q=<field>:<v1>,<v2>&page=<i>&size=<n>`` -- paginated
  metadata search. Multiple ``q`` parameters are ANDed; comma-separated values
  within one ``q`` are ORed. ``size`` is capped at 500 by the server.
* the standardised morphology file, served as a static asset at
  ``/dableFiles/<archive>/CNG version/<neuron_name>.CNG.swc``.

The CNG-standardised variant is the one to use for modelling: it has been
re-rooted, de-duplicated and unit-checked by the NeuroMorpho curation pipeline,
which matters when a population sweep must run unattended over hundreds of files
contributed by dozens of labs.

Network access is deliberately confined to this module so the rest of the package
is unit-testable offline.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import quote

import requests

from .swc_cable import SWCNeuron, load_swc

__all__ = [
    "NeuroMorphoClient",
    "NeuroMorphoQuery",
    "HUMAN_CORTICAL_QUERY",
    "MOUSE_CORTICAL_QUERY",
]

LOG = logging.getLogger(__name__)

BASE_URL = "https://neuromorpho.org/api"
DABLE_URL = "https://neuromorpho.org/dableFiles"
#: Sent because the service asks automated clients to identify themselves.
USER_AGENT = "morph_stim/0.1 (research use; https://neuromorpho.org/api.jsp)"


@dataclass(frozen=True)
class NeuroMorphoQuery:
    """A filter over NeuroMorpho metadata fields.

    Each attribute maps to one ``q=<field>:<values>`` clause. Values within a
    field are ORed by the server; fields are ANDed. Empty tuples are omitted.

    Attributes:
        species: e.g. ``("human",)`` or ``("mouse",)``.
        brain_region: e.g. ``("neocortex",)``. NeuroMorpho's region vocabulary is
            hierarchical and inconsistently populated across archives, so filter
            broadly here and refine on the returned metadata.
        cell_type: e.g. ``("pyramidal",)``, ``("interneuron",)``.
        experiment_condition: e.g. ``("control",)``, ``("alzheimer's disease",)``.
            Used for the diseased/aged comparison arm.
        archive: Restrict to specific contributing archives.
        extra: Any further ``field: values`` clauses.
    """

    species: tuple[str, ...] = ()
    brain_region: tuple[str, ...] = ()
    cell_type: tuple[str, ...] = ()
    experiment_condition: tuple[str, ...] = ()
    archive: tuple[str, ...] = ()
    extra: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def clauses(self) -> list[str]:
        """Render the query as a list of ``field:v1,v2`` strings."""
        named = (
            ("species", self.species),
            ("brain_region", self.brain_region),
            ("cell_type", self.cell_type),
            ("experiment_condition", self.experiment_condition),
            ("archive", self.archive),
        )
        out = [f"{k}:{','.join(v)}" for k, v in named if v]
        out += [f"{k}:{','.join(v)}" for k, v in self.extra if v]
        return out

    def slug(self) -> str:
        """Filesystem-safe identifier for caching this query's results."""
        raw = "__".join(self.clauses()) or "all"
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)[:120]


#: The two primary cohorts. Cortical pyramidal cells and interneurons, control
#: condition, in the two species with enough human reconstructions to compare.
HUMAN_CORTICAL_QUERY = NeuroMorphoQuery(
    species=("human",),
    brain_region=("neocortex",),
    cell_type=("pyramidal", "interneuron"),
)
MOUSE_CORTICAL_QUERY = NeuroMorphoQuery(
    species=("mouse",),
    brain_region=("neocortex",),
    cell_type=("pyramidal", "interneuron"),
)


@dataclass
class NeuroMorphoClient:
    """Thin, polite HTTP client for NeuroMorpho.Org.

    Attributes:
        cache_dir: Root for cached metadata JSON and SWC files.
        page_size: Records per request (server maximum is 500).
        delay_s: Sleep between requests. The service is a shared academic
            resource; do not remove this.
        timeout_s: Per-request timeout.
        max_retries: Retries per request on transport errors and 5xx.
        verify: Passed to ``requests``; leave ``True``.
    """

    cache_dir: Path = Path("data/neuromorpho")
    page_size: int = 100
    delay_s: float = 0.5
    timeout_s: float = 60.0
    max_retries: int = 3
    verify: bool = True
    _session: requests.Session = field(default_factory=requests.Session, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self._session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})

    # ------------------------------------------------------------------ plumbing
    def _get(self, url: str, *, as_json: bool = True) -> Any:
        """GET with retries and exponential backoff.

        Raises:
            requests.HTTPError: On a non-retryable or persistent HTTP error.
        """
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self._session.get(url, timeout=self.timeout_s, verify=self.verify)
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"{resp.status_code} from {url}", response=resp)
                resp.raise_for_status()
                time.sleep(self.delay_s)
                return resp.json() if as_json else resp.text
            except (requests.RequestException, json.JSONDecodeError) as exc:
                last = exc
                wait = self.delay_s * (2**attempt)
                LOG.warning("GET %s failed (attempt %d/%d): %s", url, attempt + 1, self.max_retries, exc)
                time.sleep(wait)
        raise requests.HTTPError(f"giving up on {url}") from last

    def health(self) -> dict[str, Any]:
        """Return the service health document (``/api/health``)."""
        return self._get(f"{BASE_URL}/health")

    # ------------------------------------------------------------------- search
    def iter_neurons(
        self, query: NeuroMorphoQuery, *, limit: int | None = None
    ) -> Iterator[dict[str, Any]]:
        """Yield metadata records matching ``query``, following pagination.

        Args:
            query: Metadata filter.
            limit: Stop after this many records; ``None`` for all.

        Yields:
            One metadata dict per reconstruction, as returned by the API.
        """
        qs = "&".join(f"q={quote(c, safe=':,')}" for c in query.clauses())
        page = 0
        yielded = 0
        while True:
            url = f"{BASE_URL}/neuron/select?{qs}&page={page}&size={self.page_size}"
            payload = self._get(url)
            records = _extract_records(payload)
            if not records:
                return
            for rec in records:
                yield rec
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
            info = payload.get("page", {}) if isinstance(payload, dict) else {}
            total_pages = int(info.get("totalPages", page + 1))
            page += 1
            if page >= total_pages:
                return

    def search(
        self, query: NeuroMorphoQuery, *, limit: int | None = None, use_cache: bool = True
    ) -> list[dict[str, Any]]:
        """Collect all records for ``query``, caching the result as JSON.

        Args:
            query: Metadata filter.
            limit: Maximum records.
            use_cache: Reuse a previously written cache file if present.

        Returns:
            List of metadata dicts.
        """
        cache = self.cache_dir / "metadata" / f"{query.slug()}.json"
        if use_cache and cache.exists():
            LOG.info("using cached metadata %s", cache)
            return json.loads(cache.read_text(encoding="utf-8"))
        records = list(self.iter_neurons(query, limit=limit))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(records, indent=1), encoding="utf-8")
        LOG.info("cached %d records to %s", len(records), cache)
        return records

    # --------------------------------------------------------------- morphology
    @staticmethod
    def swc_url(record: dict[str, Any]) -> str:
        """Build the CNG-standardised SWC URL for a metadata record.

        The archive directory on the file server is the archive name lowercased
        with spaces removed.

        Args:
            record: A metadata dict containing ``archive`` and ``neuron_name``.

        Returns:
            Absolute URL of the ``.CNG.swc`` file.

        Raises:
            KeyError: If the record lacks the required fields.
        """
        archive = str(record["archive"]).lower().replace(" ", "")
        name = str(record["neuron_name"])
        return f"{DABLE_URL}/{quote(archive)}/CNG%20version/{quote(name)}.CNG.swc"

    def fetch_swc(self, record: dict[str, Any], *, use_cache: bool = True) -> Path:
        """Download one CNG SWC file into the cache and return its path.

        Args:
            record: Metadata dict for the reconstruction.
            use_cache: Skip the download if the file already exists.

        Returns:
            Path of the local ``.swc`` file.
        """
        out = self.cache_dir / "swc" / f"{record['neuron_name']}.CNG.swc"
        if use_cache and out.exists() and out.stat().st_size > 0:
            return out
        text = self._get(self.swc_url(record), as_json=False)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        return out

    def load_population(
        self,
        query: NeuroMorphoQuery,
        *,
        limit: int | None = None,
        metadata_fields: Sequence[str] = (
            "species",
            "brain_region",
            "cell_type",
            "archive",
            "experiment_condition",
            "age_classification",
            "gender",
            "neuron_id",
        ),
        skip_errors: bool = True,
    ) -> list[SWCNeuron]:
        """Search, download and parse a whole cohort.

        Args:
            query: Metadata filter.
            limit: Maximum reconstructions.
            metadata_fields: Record fields copied onto each :class:`SWCNeuron`.
            skip_errors: Log and skip reconstructions that fail to download or
                parse instead of raising. Recommended for population runs.

        Returns:
            Parsed morphologies with provenance attached.
        """
        neurons: list[SWCNeuron] = []
        for rec in self.search(query, limit=limit):
            try:
                path = self.fetch_swc(rec)
                meta = {k: _flatten(rec.get(k)) for k in metadata_fields}
                meta["layer"] = _infer_layer(rec)
                neurons.append(load_swc(path, name=str(rec.get("neuron_name")), **meta))
            except Exception as exc:  # noqa: BLE001 - population robustness
                if not skip_errors:
                    raise
                LOG.warning("skipping %s: %s", rec.get("neuron_name"), exc)
        return neurons


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    """Pull the neuron list out of a HAL response, tolerating shape changes."""
    if not isinstance(payload, dict):
        return []
    for container in (payload.get("_embedded"), payload):
        if isinstance(container, dict):
            for key in ("neuronResources", "neurons", "_embedded"):
                value = container.get(key)
                if isinstance(value, list):
                    return [v for v in value if isinstance(v, dict)]
    return []


def _flatten(value: Any) -> Any:
    """Collapse the single-element lists NeuroMorpho returns for many fields."""
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        return "|".join(str(v) for v in value)
    return value


def _infer_layer(record: dict[str, Any]) -> str | None:
    """Best-effort cortical layer from the free-text ``brain_region`` field.

    NeuroMorpho has no dedicated layer column; layer is usually the third
    element of the ``brain_region`` hierarchy (e.g.
    ``["neocortex", "frontal", "layer 5"]``). Returns ``None`` when absent, and
    the README treats missing layer as a covariate to be imputed or excluded
    rather than guessed.
    """
    region = record.get("brain_region")
    candidates = region if isinstance(region, list) else [region]
    for item in candidates:
        text = str(item).lower()
        if "layer" in text or text.strip() in {"l1", "l2", "l3", "l4", "l5", "l6", "l2/3"}:
            return text.strip()
    return None
