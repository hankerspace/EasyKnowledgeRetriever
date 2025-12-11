import asyncio
import json
import os
import re
import datetime
from datetime import timezone
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TypeVar, Union, final
import numpy as np
import configparser
import ssl
import itertools

from ..types import KnowledgeGraph, KnowledgeGraphNode, KnowledgeGraphEdge

from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
)

from .base import BaseGraphStorage
from ..kv_storage.base import BaseKVStorage, DocProcessingStatus, DocStatus, DocStatusStorage
from ..vector_storage.base import BaseVectorStorage
from ..namespace import NameSpace, is_namespace
from easy_knowledge_retriever.utils.logger import logger
from easy_knowledge_retriever.kg.concurrency import get_data_init_lock
from easy_knowledge_retriever.constants import (
    DEFAULT_EMBEDDING_BATCH_NUM,
    DEFAULT_MAX_GRAPH_NODES,
)


import asyncpg  # type: ignore
from asyncpg import Pool  # type: ignore

T = TypeVar("T")



from ..postgres_database import PostgreSQLDB, ClientManager

class PGGraphQueryException(Exception):
    """Exception for the AGE queries."""

    def __init__(self, exception: Union[str, dict[str, Any]]) -> None:
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details




@final
@dataclass
class PGGraphStorage(BaseGraphStorage):
    db: PostgreSQLDB = field(default=None)
    max_graph_nodes: int = field(default=DEFAULT_MAX_GRAPH_NODES)

    def __post_init__(self):
        # Graph name will be dynamically generated in initialize() based on workspace
        self.db: PostgreSQLDB | None = None

    def _get_workspace_graph_name(self) -> str:
        """
        Generate graph name based on workspace and namespace for data isolation.
        Rules:
        - If workspace is empty or "default": graph_name = namespace
        - If workspace has other value: graph_name = workspace_namespace

        Args:
            None

        Returns:
            str: The graph name for the current workspace
        """
        workspace = self.workspace
        namespace = self.namespace

        if workspace and workspace.strip() and workspace.strip().lower() != "default":
            # Ensure names comply with PostgreSQL identifier specifications
            safe_workspace = re.sub(r"[^a-zA-Z0-9_]", "_", workspace.strip())
            safe_namespace = re.sub(r"[^a-zA-Z0-9_]", "_", namespace)
            return f"{safe_workspace}_{safe_namespace}"
        else:
            # When the workspace is "default", use the namespace directly (for backward compatibility with legacy implementations)
            return re.sub(r"[^a-zA-Z0-9_]", "_", namespace)

    @staticmethod
    def _normalize_node_id(node_id: str) -> str:
        """
        Normalize node ID to ensure special characters are properly handled in Cypher queries.

        Args:
            node_id: The original node ID

        Returns:
            Normalized node ID suitable for Cypher queries
        """
        # Escape backslashes
        normalized_id = node_id
        normalized_id = normalized_id.replace("\\", "\\\\")
        normalized_id = normalized_id.replace('"', '\\"')
        return normalized_id

    async def initialize(self):
        async with get_data_init_lock():
            if self.db is None:
                self.db = await ClientManager.get_client()

            # Implement workspace priority: PostgreSQLDB.workspace > self.workspace > "default"
            if self.db.workspace:
                # Use PostgreSQLDB's workspace (highest priority)
                self.workspace = self.db.workspace
            elif hasattr(self, "workspace") and self.workspace:
                # Use storage class's workspace (medium priority)
                pass
            else:
                # Use "default" for compatibility (lowest priority)
                self.workspace = "default"

            # Dynamically generate graph name based on workspace
            self.graph_name = self._get_workspace_graph_name()

            # Log the graph initialization for debugging
            logger.info(
                f"[{self.workspace}] PostgreSQL Graph initialized: graph_name='{self.graph_name}'"
            )

            # Create AGE extension and configure graph environment once at initialization
            async with self.db.pool.acquire() as connection:
                # First ensure AGE extension is created
                await PostgreSQLDB.configure_age_extension(connection)

            # Execute each statement separately and ignore errors
            queries = [
                f"SELECT create_graph('{self.graph_name}')",
                f"SELECT create_vlabel('{self.graph_name}', 'base');",
                f"SELECT create_elabel('{self.graph_name}', 'DIRECTED');",
                # f'CREATE INDEX CONCURRENTLY vertex_p_idx ON {self.graph_name}."_ag_label_vertex" (id)',
                f'CREATE INDEX CONCURRENTLY vertex_idx_node_id ON {self.graph_name}."_ag_label_vertex" (ag_catalog.agtype_access_operator(properties, \'"entity_id"\'::agtype))',
                # f'CREATE INDEX CONCURRENTLY edge_p_idx ON {self.graph_name}."_ag_label_edge" (id)',
                f'CREATE INDEX CONCURRENTLY edge_sid_idx ON {self.graph_name}."_ag_label_edge" (start_id)',
                f'CREATE INDEX CONCURRENTLY edge_eid_idx ON {self.graph_name}."_ag_label_edge" (end_id)',
                f'CREATE INDEX CONCURRENTLY edge_seid_idx ON {self.graph_name}."_ag_label_edge" (start_id,end_id)',
                f'CREATE INDEX CONCURRENTLY directed_p_idx ON {self.graph_name}."DIRECTED" (id)',
                f'CREATE INDEX CONCURRENTLY directed_eid_idx ON {self.graph_name}."DIRECTED" (end_id)',
                f'CREATE INDEX CONCURRENTLY directed_sid_idx ON {self.graph_name}."DIRECTED" (start_id)',
                f'CREATE INDEX CONCURRENTLY directed_seid_idx ON {self.graph_name}."DIRECTED" (start_id,end_id)',
                f'CREATE INDEX CONCURRENTLY entity_p_idx ON {self.graph_name}."base" (id)',
                f'CREATE INDEX CONCURRENTLY entity_idx_node_id ON {self.graph_name}."base" (ag_catalog.agtype_access_operator(properties, \'"entity_id"\'::agtype))',
                f'CREATE INDEX CONCURRENTLY entity_node_id_gin_idx ON {self.graph_name}."base" using gin(properties)',
                f'ALTER TABLE {self.graph_name}."DIRECTED" CLUSTER ON directed_sid_idx',
            ]

            for query in queries:
                # Use the new flag to silently ignore "already exists" errors
                # at the source, preventing log spam.
                await self.db.execute(
                    query,
                    upsert=True,
                    ignore_if_exists=True,  # Pass the new flag
                    with_age=True,
                    graph_name=self.graph_name,
                )

    async def finalize(self):
        if self.db is not None:
            await ClientManager.release_client(self.db)
            self.db = None

    async def index_done_callback(self) -> None:
        # PG handles persistence automatically
        pass

    @staticmethod
    def _record_to_dict(record: asyncpg.Record) -> dict[str, Any]:
        """
        Convert a record returned from an age query to a dictionary

        Args:
            record (): a record from an age query result

        Returns:
            dict[str, Any]: a dictionary representation of the record where
                the dictionary key is the field name and the value is the
                value converted to a python type
        """

        @staticmethod
        def parse_agtype_string(agtype_str: str) -> tuple[str, str]:
            """
            Parse agtype string precisely, separating JSON content and type identifier

            Args:
                agtype_str: String like '{"json": "content"}::vertex'

            Returns:
                (json_content, type_identifier)
            """
            if not isinstance(agtype_str, str) or "::" not in agtype_str:
                return agtype_str, ""

            # Find the last :: from the right, which is the start of type identifier
            last_double_colon = agtype_str.rfind("::")

            if last_double_colon == -1:
                return agtype_str, ""

            # Separate JSON content and type identifier
            json_content = agtype_str[:last_double_colon]
            type_identifier = agtype_str[last_double_colon + 2 :]

            return json_content, type_identifier

        @staticmethod
        def safe_json_parse(json_str: str, context: str = "") -> dict:
            """
            Safe JSON parsing with simplified error logging
            """
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed ({context}): {e}")
                logger.error(f"Raw data (first 100 chars): {repr(json_str[:100])}")
                logger.error(f"Error position: line {e.lineno}, column {e.colno}")
                return None

        # result holder
        d = {}

        # prebuild a mapping of vertex_id to vertex mappings to be used
        # later to build edges
        vertices = {}

        # First pass: preprocess vertices
        for k in record.keys():
            v = record[k]
            if isinstance(v, str) and "::" in v:
                if v.startswith("[") and v.endswith("]"):
                    # Handle vertex arrays
                    json_content, type_id = parse_agtype_string(v)
                    if type_id == "vertex":
                        vertexes = safe_json_parse(
                            json_content, f"vertices array for {k}"
                        )
                        if vertexes:
                            for vertex in vertexes:
                                vertices[vertex["id"]] = vertex.get("properties")
                else:
                    # Handle single vertex
                    json_content, type_id = parse_agtype_string(v)
                    if type_id == "vertex":
                        vertex = safe_json_parse(json_content, f"single vertex for {k}")
                        if vertex:
                            vertices[vertex["id"]] = vertex.get("properties")

        # Second pass: process all fields
        for k in record.keys():
            v = record[k]
            if isinstance(v, str) and "::" in v:
                if v.startswith("[") and v.endswith("]"):
                    # Handle array types
                    json_content, type_id = parse_agtype_string(v)
                    if type_id in ["vertex", "edge"]:
                        parsed_data = safe_json_parse(
                            json_content, f"array {type_id} for field {k}"
                        )
                        d[k] = parsed_data if parsed_data is not None else None
                    else:
                        logger.warning(f"Unknown array type: {type_id}")
                        d[k] = None
                else:
                    # Handle single objects
                    json_content, type_id = parse_agtype_string(v)
                    if type_id in ["vertex", "edge"]:
                        parsed_data = safe_json_parse(
                            json_content, f"single {type_id} for field {k}"
                        )
                        d[k] = parsed_data if parsed_data is not None else None
                    else:
                        # May be other types of agtype data, keep as is
                        d[k] = v
            else:
                d[k] = v  # Keep as string

        return d

    @staticmethod
    def _format_properties(
        properties: dict[str, Any], _id: Union[str, None] = None
    ) -> str:
        """
        Convert a dictionary of properties to a string representation that
        can be used in a cypher query insert/merge statement.

        Args:
            properties (dict[str,str]): a dictionary containing node/edge properties
            _id (Union[str, None]): the id of the node or None if none exists

        Returns:
            str: the properties dictionary as a properly formatted string
        """
        props = []
        # wrap property key in backticks to escape
        for k, v in properties.items():
            prop = f"`{k}`: {json.dumps(v)}"
            props.append(prop)
        if _id is not None and "id" not in properties:
            props.append(
                f"id: {json.dumps(_id)}" if isinstance(_id, str) else f"id: {_id}"
            )
        return "{" + ", ".join(props) + "}"

    async def _query(
        self,
        query: str,
        readonly: bool = True,
        upsert: bool = False,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Query the graph by taking a cypher query, converting it to an
        age compatible query, executing it and converting the result

        Args:
            query (str): a cypher query to be executed

        Returns:
            list[dict[str, Any]]: a list of dictionaries containing the result set
        """
        try:
            if readonly:
                data = await self.db.query(
                    query,
                    list(params.values()) if params else None,
                    multirows=True,
                    with_age=True,
                    graph_name=self.graph_name,
                )
            else:
                data = await self.db.execute(
                    query,
                    upsert=upsert,
                    with_age=True,
                    graph_name=self.graph_name,
                )

        except Exception as e:
            raise PGGraphQueryException(
                {
                    "message": f"Error executing graph query: {query}",
                    "wrapped": query,
                    "detail": repr(e),
                    "error_type": e.__class__.__name__,
                }
            ) from e

        if data is None:
            result = []
        # decode records
        else:
            result = [self._record_to_dict(d) for d in data]

        return result

    async def has_node(self, node_id: str) -> bool:
        query = f"""
            SELECT EXISTS (
              SELECT 1
              FROM {self.graph_name}.base
              WHERE ag_catalog.agtype_access_operator(
                      VARIADIC ARRAY[properties, '"entity_id"'::agtype]
                    ) = (to_json($1::text)::text)::agtype
              LIMIT 1
            ) AS node_exists;
        """

        params = {"node_id": node_id}
        row = (await self._query(query, params=params))[0]
        return bool(row["node_exists"])

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        query = f"""
            WITH a AS (
              SELECT id AS vid
              FROM {self.graph_name}.base
              WHERE ag_catalog.agtype_access_operator(
                      VARIADIC ARRAY[properties, '"entity_id"'::agtype]
                    ) = (to_json($1::text)::text)::agtype
            ),
            b AS (
              SELECT id AS vid
              FROM {self.graph_name}.base
              WHERE ag_catalog.agtype_access_operator(
                      VARIADIC ARRAY[properties, '"entity_id"'::agtype]
                    ) = (to_json($2::text)::text)::agtype
            )
            SELECT EXISTS (
              SELECT 1
              FROM {self.graph_name}."DIRECTED" d
              JOIN a ON d.start_id = a.vid
              JOIN b ON d.end_id   = b.vid
              LIMIT 1
            )
            OR EXISTS (
              SELECT 1
              FROM {self.graph_name}."DIRECTED" d
              JOIN a ON d.end_id   = a.vid
              JOIN b ON d.start_id = b.vid
              LIMIT 1
            ) AS edge_exists;
        """
        params = {
            "source_node_id": source_node_id,
            "target_node_id": target_node_id,
        }
        row = (await self._query(query, params=params))[0]
        return bool(row["edge_exists"])

    async def get_node(self, node_id: str) -> dict[str, str] | None:
        """Get node by its label identifier, return only node properties"""

        result = await self.get_nodes_batch(node_ids=[node_id])
        if result and node_id in result:
            return result[node_id]
        return None

    async def node_degree(self, node_id: str) -> int:
        result = await self.node_degrees_batch(node_ids=[node_id])
        if result and node_id in result:
            return result[node_id]

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        result = await self.edge_degrees_batch(edges=[(src_id, tgt_id)])
        if result and (src_id, tgt_id) in result:
            return result[(src_id, tgt_id)]

    async def get_edge(
        self, source_node_id: str, target_node_id: str
    ) -> dict[str, str] | None:
        """Get edge properties between two nodes"""
        result = await self.get_edges_batch(
            [{"src": source_node_id, "tgt": target_node_id}]
        )
        if result and (source_node_id, target_node_id) in result:
            return result[(source_node_id, target_node_id)]
        return None

    async def get_node_edges(self, source_node_id: str) -> list[tuple[str, str]] | None:
        """
        Retrieves all edges (relationships) for a particular node identified by its label.
        :return: list of dictionaries containing edge information
        """
        label = self._normalize_node_id(source_node_id)

        query = """SELECT * FROM cypher('%s', $$
                      MATCH (n:base {entity_id: "%s"})
                      OPTIONAL MATCH (n)-[]-(connected:base)
                      RETURN n.entity_id AS source_id, connected.entity_id AS connected_id
                    $$) AS (source_id text, connected_id text)""" % (
            self.graph_name,
            label,
        )

        results = await self._query(query)
        edges = []
        for record in results:
            source_id = record["source_id"]
            connected_id = record["connected_id"]

            if source_id and connected_id:
                edges.append((source_id, connected_id))

        return edges

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PGGraphQueryException,)),
    )
    async def upsert_node(self, node_id: str, node_data: dict[str, str]) -> None:
        """
        Upsert a node in the Neo4j database.

        Args:
            node_id: The unique identifier for the node (used as label)
            node_data: Dictionary of node properties
        """
        if "entity_id" not in node_data:
            raise ValueError(
                "PostgreSQL: node properties must contain an 'entity_id' field"
            )

        label = self._normalize_node_id(node_id)
        properties = self._format_properties(node_data)

        query = """SELECT * FROM cypher('%s', $$
                     MERGE (n:base {entity_id: "%s"})
                     SET n += %s
                     RETURN n
                   $$) AS (n agtype)""" % (
            self.graph_name,
            label,
            properties,
        )

        try:
            await self._query(query, readonly=False, upsert=True)

        except Exception:
            logger.error(
                f"[{self.workspace}] POSTGRES, upsert_node error on node_id: `{node_id}`"
            )
            raise

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((PGGraphQueryException,)),
    )
    async def upsert_edge(
        self, source_node_id: str, target_node_id: str, edge_data: dict[str, str]
    ) -> None:
        """
        Upsert an edge and its properties between two nodes identified by their labels.

        Args:
            source_node_id (str): Label of the source node (used as identifier)
            target_node_id (str): Label of the target node (used as identifier)
            edge_data (dict): dictionary of properties to set on the edge
        """
        src_label = self._normalize_node_id(source_node_id)
        tgt_label = self._normalize_node_id(target_node_id)
        edge_properties = self._format_properties(edge_data)

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (source:base {entity_id: "%s"})
                     WITH source
                     MATCH (target:base {entity_id: "%s"})
                     MERGE (source)-[r:DIRECTED]-(target)
                     SET r += %s
                     SET r += %s
                     RETURN r
                   $$) AS (r agtype)""" % (
            self.graph_name,
            src_label,
            tgt_label,
            edge_properties,
            edge_properties,  # https://github.com/HKUDS/EasyKnowledgeRetriever/issues/1438#issuecomment-2826000195
        )

        try:
            await self._query(query, readonly=False, upsert=True)

        except Exception:
            logger.error(
                f"[{self.workspace}] POSTGRES, upsert_edge error on edge: `{source_node_id}`-`{target_node_id}`"
            )
            raise

    async def delete_node(self, node_id: str) -> None:
        """
        Delete a node from the graph.

        Args:
            node_id (str): The ID of the node to delete.
        """
        label = self._normalize_node_id(node_id)

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:base {entity_id: "%s"})
                     DETACH DELETE n
                   $$) AS (n agtype)""" % (self.graph_name, label)

        try:
            await self._query(query, readonly=False)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error during node deletion: {e}")
            raise

    async def remove_nodes(self, node_ids: list[str]) -> None:
        """
        Remove multiple nodes from the graph.

        Args:
            node_ids (list[str]): A list of node IDs to remove.
        """
        node_ids = [self._normalize_node_id(node_id) for node_id in node_ids]
        node_id_list = ", ".join([f'"{node_id}"' for node_id in node_ids])

        query = """SELECT * FROM cypher('%s', $$
                     MATCH (n:base)
                     WHERE n.entity_id IN [%s]
                     DETACH DELETE n
                   $$) AS (n agtype)""" % (self.graph_name, node_id_list)

        try:
            await self._query(query, readonly=False)
        except Exception as e:
            logger.error(f"[{self.workspace}] Error during node removal: {e}")
            raise

    async def remove_edges(self, edges: list[tuple[str, str]]) -> None:
        """
        Remove multiple edges from the graph.

        Args:
            edges (list[tuple[str, str]]): A list of edges to remove, where each edge is a tuple of (source_node_id, target_node_id).
        """
        for source, target in edges:
            src_label = self._normalize_node_id(source)
            tgt_label = self._normalize_node_id(target)

            query = """SELECT * FROM cypher('%s', $$
                         MATCH (a:base {entity_id: "%s"})-[r]-(b:base {entity_id: "%s"})
                         DELETE r
                       $$) AS (r agtype)""" % (self.graph_name, src_label, tgt_label)

            try:
                await self._query(query, readonly=False)
                logger.debug(
                    f"[{self.workspace}] Deleted edge from '{source}' to '{target}'"
                )
            except Exception as e:
                logger.error(f"[{self.workspace}] Error during edge deletion: {str(e)}")
                raise

    async def get_nodes_batch(
        self, node_ids: list[str], batch_size: int = 1000
    ) -> dict[str, dict]:
        """
        Retrieve multiple nodes in one query using UNWIND.

        Args:
            node_ids: List of node entity IDs to fetch.
            batch_size: Batch size for the query

        Returns:
            A dictionary mapping each node_id to its node data (or None if not found).
        """
        if not node_ids:
            return {}

        seen: set[str] = set()
        unique_ids: list[str] = []
        lookup: dict[str, str] = {}
        requested: set[str] = set()
        for nid in node_ids:
            if nid not in seen:
                seen.add(nid)
                unique_ids.append(nid)
            requested.add(nid)
            lookup[nid] = nid
            lookup[self._normalize_node_id(nid)] = nid

        # Build result dictionary
        nodes_dict = {}

        for i in range(0, len(unique_ids), batch_size):
            batch = unique_ids[i : i + batch_size]

            query = f"""
                WITH input(v, ord) AS (
                  SELECT v, ord
                  FROM unnest($1::text[]) WITH ORDINALITY AS t(v, ord)
                ),
                ids(node_id, ord) AS (
                  SELECT (to_json(v)::text)::agtype AS node_id, ord
                  FROM input
                )
                SELECT i.node_id::text AS node_id,
                       b.properties
                FROM {self.graph_name}.base AS b
                JOIN ids i
                  ON ag_catalog.agtype_access_operator(
                       VARIADIC ARRAY[b.properties, '"entity_id"'::agtype]
                     ) = i.node_id
                ORDER BY i.ord;
            """

            results = await self._query(query, params={"ids": batch})

            for result in results:
                if result["node_id"] and result["properties"]:
                    node_dict = result["properties"]

                    # Process string result, parse it to JSON dictionary
                    if isinstance(node_dict, str):
                        try:
                            node_dict = json.loads(node_dict)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"[{self.workspace}] Failed to parse node string in batch: {node_dict}"
                            )

                    node_key = result["node_id"]
                    original_key = lookup.get(node_key)
                    if original_key is None:
                        logger.warning(
                            f"[{self.workspace}] Node {node_key} not found in lookup map"
                        )
                        original_key = node_key
                    if original_key in requested:
                        nodes_dict[original_key] = node_dict

        return nodes_dict

    async def node_degrees_batch(
        self, node_ids: list[str], batch_size: int = 500
    ) -> dict[str, int]:
        """
        Retrieve the degree for multiple nodes in a single query using UNWIND.
        Calculates the total degree by counting distinct relationships.
        Uses separate queries for outgoing and incoming edges.

        Args:
            node_ids: List of node labels (entity_id values) to look up.
            batch_size: Batch size for the query

        Returns:
            A dictionary mapping each node_id to its degree (total number of relationships).
            If a node is not found, its degree will be set to 0.
        """
        if not node_ids:
            return {}

        seen: set[str] = set()
        unique_ids: list[str] = []
        lookup: dict[str, str] = {}
        requested: set[str] = set()
        for nid in node_ids:
            if nid not in seen:
                seen.add(nid)
                unique_ids.append(nid)
            requested.add(nid)
            lookup[nid] = nid
            lookup[self._normalize_node_id(nid)] = nid

        out_degrees = {}
        in_degrees = {}

        for i in range(0, len(unique_ids), batch_size):
            batch = unique_ids[i : i + batch_size]

            query = f"""
                    WITH input(v, ord) AS (
                      SELECT v, ord
                      FROM unnest($1::text[]) WITH ORDINALITY AS t(v, ord)
                    ),
                    ids(node_id, ord) AS (
                      SELECT (to_json(v)::text)::agtype AS node_id, ord
                      FROM input
                    ),
                    vids AS (
                      SELECT b.id AS vid, i.node_id, i.ord
                      FROM {self.graph_name}.base AS b
                      JOIN ids i
                        ON ag_catalog.agtype_access_operator(
                             VARIADIC ARRAY[b.properties, '"entity_id"'::agtype]
                           ) = i.node_id
                    ),
                    deg_out AS (
                      SELECT d.start_id AS vid, COUNT(*)::bigint AS out_degree
                      FROM {self.graph_name}."DIRECTED" AS d
                      JOIN vids v ON v.vid = d.start_id
                      GROUP BY d.start_id
                    ),
                    deg_in AS (
                      SELECT d.end_id AS vid, COUNT(*)::bigint AS in_degree
                      FROM {self.graph_name}."DIRECTED" AS d
                      JOIN vids v ON v.vid = d.end_id
                      GROUP BY d.end_id
                    )
                    SELECT v.node_id::text AS node_id,
                           COALESCE(o.out_degree, 0) AS out_degree,
                           COALESCE(n.in_degree, 0)  AS in_degree
                    FROM vids v
                    LEFT JOIN deg_out o ON o.vid = v.vid
                    LEFT JOIN deg_in  n ON n.vid = v.vid
                    ORDER BY v.ord;
                """

            combined_results = await self._query(query, params={"ids": batch})

            for row in combined_results:
                node_id = row["node_id"]
                if not node_id:
                    continue
                node_key = node_id
                original_key = lookup.get(node_key)
                if original_key is None:
                    logger.warning(
                        f"[{self.workspace}] Node {node_key} not found in lookup map"
                    )
                    original_key = node_key
                if original_key in requested:
                    out_degrees[original_key] = int(row.get("out_degree", 0) or 0)
                    in_degrees[original_key] = int(row.get("in_degree", 0) or 0)

        degrees_dict = {}
        for node_id in node_ids:
            out_degree = out_degrees.get(node_id, 0)
            in_degree = in_degrees.get(node_id, 0)
            degrees_dict[node_id] = out_degree + in_degree

        return degrees_dict

    async def edge_degrees_batch(
        self, edges: list[tuple[str, str]]
    ) -> dict[tuple[str, str], int]:
        """
        Calculate the combined degree for each edge (sum of the source and target node degrees)
        in batch using the already implemented node_degrees_batch.

        Args:
            edges: List of (source_node_id, target_node_id) tuples

        Returns:
            Dictionary mapping edge tuples to their combined degrees
        """
        if not edges:
            return {}

        # Use node_degrees_batch to get all node degrees efficiently
        all_nodes = set()
        for src, tgt in edges:
            all_nodes.add(src)
            all_nodes.add(tgt)

        node_degrees = await self.node_degrees_batch(list(all_nodes))

        # Calculate edge degrees
        edge_degrees_dict = {}
        for src, tgt in edges:
            src_degree = node_degrees.get(src, 0)
            tgt_degree = node_degrees.get(tgt, 0)
            edge_degrees_dict[(src, tgt)] = src_degree + tgt_degree

        return edge_degrees_dict

    async def get_edges_batch(
        self, pairs: list[dict[str, str]], batch_size: int = 500
    ) -> dict[tuple[str, str], dict]:
        """
        Retrieve edge properties for multiple (src, tgt) pairs in one query.
        Get forward and backward edges seperately and merge them before return

        Args:
            pairs: List of dictionaries, e.g. [{"src": "node1", "tgt": "node2"}, ...]
            batch_size: Batch size for the query

        Returns:
            A dictionary mapping (src, tgt) tuples to their edge properties.
        """
        if not pairs:
            return {}

        seen = set()
        uniq_pairs: list[dict[str, str]] = []
        for p in pairs:
            s = self._normalize_node_id(p["src"])
            t = self._normalize_node_id(p["tgt"])
            key = (s, t)
            if s and t and key not in seen:
                seen.add(key)
                uniq_pairs.append(p)

        edges_dict: dict[tuple[str, str], dict] = {}

        for i in range(0, len(uniq_pairs), batch_size):
            batch = uniq_pairs[i : i + batch_size]

            pairs = [{"src": p["src"], "tgt": p["tgt"]} for p in batch]

            forward_cypher = """
                         UNWIND $pairs AS p
                         WITH p.src AS src_eid, p.tgt AS tgt_eid
                         MATCH (a:base {entity_id: src_eid})
                         MATCH (b:base {entity_id: tgt_eid})
                         MATCH (a)-[r]->(b)
                         RETURN src_eid AS source, tgt_eid AS target, properties(r) AS edge_properties"""
            backward_cypher = """
                         UNWIND $pairs AS p
                         WITH p.src AS src_eid, p.tgt AS tgt_eid
                         MATCH (a:base {entity_id: src_eid})
                         MATCH (b:base {entity_id: tgt_eid})
                         MATCH (a)<-[r]-(b)
                         RETURN src_eid AS source, tgt_eid AS target, properties(r) AS edge_properties"""

            def dollar_quote(s: str, tag_prefix="AGE"):
                s = "" if s is None else str(s)
                for i in itertools.count(1):
                    tag = f"{tag_prefix}{i}"
                    wrapper = f"${tag}$"
                    if wrapper not in s:
                        return f"{wrapper}{s}{wrapper}"

            sql_fwd = f"""
            SELECT * FROM cypher({dollar_quote(self.graph_name)}::name,
                                 {dollar_quote(forward_cypher)}::cstring,
                                 $1::agtype)
              AS (source text, target text, edge_properties agtype)
            """

            sql_bwd = f"""
            SELECT * FROM cypher({dollar_quote(self.graph_name)}::name,
                                 {dollar_quote(backward_cypher)}::cstring,
                                 $1::agtype)
              AS (source text, target text, edge_properties agtype)
            """

            pg_params = {"params": json.dumps({"pairs": pairs}, ensure_ascii=False)}

            forward_results = await self._query(sql_fwd, params=pg_params)
            backward_results = await self._query(sql_bwd, params=pg_params)

            for result in forward_results:
                if result["source"] and result["target"] and result["edge_properties"]:
                    edge_props = result["edge_properties"]

                    # Process string result, parse it to JSON dictionary
                    if isinstance(edge_props, str):
                        try:
                            edge_props = json.loads(edge_props)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"[{self.workspace}]Failed to parse edge properties string: {edge_props}"
                            )
                            continue

                    edges_dict[(result["source"], result["target"])] = edge_props

            for result in backward_results:
                if result["source"] and result["target"] and result["edge_properties"]:
                    edge_props = result["edge_properties"]

                    # Process string result, parse it to JSON dictionary
                    if isinstance(edge_props, str):
                        try:
                            edge_props = json.loads(edge_props)
                        except json.JSONDecodeError:
                            logger.warning(
                                f"[{self.workspace}] Failed to parse edge properties string: {edge_props}"
                            )
                            continue

                    edges_dict[(result["source"], result["target"])] = edge_props

        return edges_dict

    async def get_nodes_edges_batch(
        self, node_ids: list[str], batch_size: int = 500
    ) -> dict[str, list[tuple[str, str]]]:
        """
        Get all edges (both outgoing and incoming) for multiple nodes in a single batch operation.

        Args:
            node_ids: List of node IDs to get edges for
            batch_size: Batch size for the query

        Returns:
            Dictionary mapping node IDs to lists of (source, target) edge tuples
        """
        if not node_ids:
            return {}

        seen = set()
        unique_ids: list[str] = []
        for nid in node_ids:
            n = self._normalize_node_id(nid)
            if n and n not in seen:
                seen.add(n)
                unique_ids.append(n)

        edges_norm: dict[str, list[tuple[str, str]]] = {n: [] for n in unique_ids}

        for i in range(0, len(unique_ids), batch_size):
            batch = unique_ids[i : i + batch_size]
            # Format node IDs for the query
            formatted_ids = ", ".join([f'"{n}"' for n in batch])

            outgoing_query = """SELECT * FROM cypher('%s', $$
                         UNWIND [%s] AS node_id
                         MATCH (n:base {entity_id: node_id})
                         OPTIONAL MATCH (n:base)-[]->(connected:base)
                         RETURN node_id, connected.entity_id AS connected_id
                       $$) AS (node_id text, connected_id text)""" % (
                self.graph_name,
                formatted_ids,
            )

            incoming_query = """SELECT * FROM cypher('%s', $$
                         UNWIND [%s] AS node_id
                         MATCH (n:base {entity_id: node_id})
                         OPTIONAL MATCH (n:base)<-[]-(connected:base)
                         RETURN node_id, connected.entity_id AS connected_id
                       $$) AS (node_id text, connected_id text)""" % (
                self.graph_name,
                formatted_ids,
            )

            outgoing_results = await self._query(outgoing_query)
            incoming_results = await self._query(incoming_query)

            for result in outgoing_results:
                if result["node_id"] and result["connected_id"]:
                    edges_norm[result["node_id"]].append(
                        (result["node_id"], result["connected_id"])
                    )

            for result in incoming_results:
                if result["node_id"] and result["connected_id"]:
                    edges_norm[result["node_id"]].append(
                        (result["connected_id"], result["node_id"])
                    )

        out: dict[str, list[tuple[str, str]]] = {}
        for orig in node_ids:
            n = self._normalize_node_id(orig)
            out[orig] = edges_norm.get(n, [])

        return out

    async def get_all_labels(self) -> list[str]:
        """
        Get all labels (node IDs) in the graph.

        Returns:
            list[str]: A list of all labels in the graph.
        """
        query = (
            """SELECT * FROM cypher('%s', $$
                     MATCH (n:base)
                     WHERE n.entity_id IS NOT NULL
                     RETURN DISTINCT n.entity_id AS label
                     ORDER BY n.entity_id
                   $$) AS (label text)"""
            % self.graph_name
        )

        results = await self._query(query)
        labels = []
        for result in results:
            if result and isinstance(result, dict) and "label" in result:
                labels.append(result["label"])
        return labels

    async def _bfs_subgraph(
        self, node_label: str, max_depth: int, max_nodes: int
    ) -> KnowledgeGraph:
        """
        Implements a true breadth-first search algorithm for subgraph retrieval.
        This method is used as a fallback when the standard Cypher query is too slow
        or when we need to guarantee BFS ordering.

        Args:
            node_label: Label of the starting node
            max_depth: Maximum depth of the subgraph
            max_nodes: Maximum number of nodes to return

        Returns:
            KnowledgeGraph object containing nodes and edges
        """
        from collections import deque

        result = KnowledgeGraph()
        visited_nodes = set()
        visited_node_ids = set()
        visited_edges = set()
        visited_edge_pairs = set()

        # Get starting node data
        label = self._normalize_node_id(node_label)
        query = """SELECT * FROM cypher('%s', $$
                    MATCH (n:base {entity_id: "%s"})
                    RETURN id(n) as node_id, n
                  $$) AS (node_id bigint, n agtype)""" % (self.graph_name, label)

        node_result = await self._query(query)
        if not node_result or not node_result[0].get("n"):
            return result

        # Create initial KnowledgeGraphNode
        start_node_data = node_result[0]["n"]
        entity_id = start_node_data["properties"]["entity_id"]
        internal_id = str(start_node_data["id"])

        start_node = KnowledgeGraphNode(
            id=internal_id,
            labels=[entity_id],
            properties=start_node_data["properties"],
        )

        # Initialize BFS queue, each element is a tuple of (node, depth)
        queue = deque([(start_node, 0)])

        visited_nodes.add(entity_id)
        visited_node_ids.add(internal_id)
        result.nodes.append(start_node)

        result.is_truncated = False

        # BFS search main loop
        while queue:
            # Get all nodes at the current depth
            current_level_nodes = []
            current_depth = None

            # Determine current depth
            if queue:
                current_depth = queue[0][1]

            # Extract all nodes at current depth from the queue
            while queue and queue[0][1] == current_depth:
                node, depth = queue.popleft()
                if depth > max_depth:
                    continue
                current_level_nodes.append(node)

            if not current_level_nodes:
                continue

            # Check depth limit
            if current_depth > max_depth:
                continue

            # Prepare node IDs list
            node_ids = [node.labels[0] for node in current_level_nodes]
            formatted_ids = ", ".join(
                [f'"{self._normalize_node_id(node_id)}"' for node_id in node_ids]
            )

            # Construct batch query for outgoing edges
            outgoing_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                UNWIND [{formatted_ids}] AS node_id
                MATCH (n:base {{entity_id: node_id}})
                OPTIONAL MATCH (n)-[r]->(neighbor:base)
                RETURN node_id AS current_id,
                       id(n) AS current_internal_id,
                       id(neighbor) AS neighbor_internal_id,
                       neighbor.entity_id AS neighbor_id,
                       id(r) AS edge_id,
                       r,
                       neighbor,
                       true AS is_outgoing
              $$) AS (current_id text, current_internal_id bigint, neighbor_internal_id bigint,
                      neighbor_id text, edge_id bigint, r agtype, neighbor agtype, is_outgoing bool)"""

            # Construct batch query for incoming edges
            incoming_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                UNWIND [{formatted_ids}] AS node_id
                MATCH (n:base {{entity_id: node_id}})
                OPTIONAL MATCH (n)<-[r]-(neighbor:base)
                RETURN node_id AS current_id,
                       id(n) AS current_internal_id,
                       id(neighbor) AS neighbor_internal_id,
                       neighbor.entity_id AS neighbor_id,
                       id(r) AS edge_id,
                       r,
                       neighbor,
                       false AS is_outgoing
              $$) AS (current_id text, current_internal_id bigint, neighbor_internal_id bigint,
                      neighbor_id text, edge_id bigint, r agtype, neighbor agtype, is_outgoing bool)"""

            # Execute queries
            outgoing_results = await self._query(outgoing_query)
            incoming_results = await self._query(incoming_query)

            # Combine results
            neighbors = outgoing_results + incoming_results

            # Create mapping from node ID to node object
            node_map = {node.labels[0]: node for node in current_level_nodes}

            # Process all results in a single loop
            for record in neighbors:
                if not record.get("neighbor") or not record.get("r"):
                    continue

                # Get current node information
                current_entity_id = record["current_id"]
                current_node = node_map[current_entity_id]

                # Get neighbor node information
                neighbor_entity_id = record["neighbor_id"]
                neighbor_internal_id = str(record["neighbor_internal_id"])
                is_outgoing = record["is_outgoing"]

                # Determine edge direction
                if is_outgoing:
                    source_id = current_node.id
                    target_id = neighbor_internal_id
                else:
                    source_id = neighbor_internal_id
                    target_id = current_node.id

                if not neighbor_entity_id:
                    continue

                # Get edge and node information
                b_node = record["neighbor"]
                rel = record["r"]
                edge_id = str(record["edge_id"])

                # Create neighbor node object
                neighbor_node = KnowledgeGraphNode(
                    id=neighbor_internal_id,
                    labels=[neighbor_entity_id],
                    properties=b_node["properties"],
                )

                # Sort entity_ids to ensure (A,B) and (B,A) are treated as the same edge
                sorted_pair = tuple(sorted([current_entity_id, neighbor_entity_id]))

                # Create edge object
                edge = KnowledgeGraphEdge(
                    id=edge_id,
                    type=rel["label"],
                    source=source_id,
                    target=target_id,
                    properties=rel["properties"],
                )

                if neighbor_internal_id in visited_node_ids:
                    # Add backward edge if neighbor node is already visited
                    if (
                        edge_id not in visited_edges
                        and sorted_pair not in visited_edge_pairs
                    ):
                        result.edges.append(edge)
                        visited_edges.add(edge_id)
                        visited_edge_pairs.add(sorted_pair)
                else:
                    if len(visited_node_ids) < max_nodes and current_depth < max_depth:
                        # Add new node to result and queue
                        result.nodes.append(neighbor_node)
                        visited_nodes.add(neighbor_entity_id)
                        visited_node_ids.add(neighbor_internal_id)

                        # Add node to queue with incremented depth
                        queue.append((neighbor_node, current_depth + 1))

                        # Add forward edge
                        if (
                            edge_id not in visited_edges
                            and sorted_pair not in visited_edge_pairs
                        ):
                            result.edges.append(edge)
                            visited_edges.add(edge_id)
                            visited_edge_pairs.add(sorted_pair)
                    else:
                        if current_depth < max_depth:
                            result.is_truncated = True

        return result

    async def get_knowledge_graph(
        self,
        node_label: str,
        max_depth: int = 3,
        max_nodes: int = None,
    ) -> KnowledgeGraph:
        """
        Retrieve a connected subgraph of nodes where the label includes the specified `node_label`.

        Args:
            node_label: Label of the starting node, * means all nodes
            max_depth: Maximum depth of the subgraph, Defaults to 3
            max_nodes: Maximum nodes to return, Defaults to self.max_graph_nodes

        Returns:
            KnowledgeGraph object containing nodes and edges, with an is_truncated flag
            indicating whether the graph was truncated due to max_nodes limit
        """
        # Use self.max_graph_nodes as default if max_nodes is None
        if max_nodes is None:
            max_nodes = self.max_graph_nodes
        else:
            # Limit max_nodes to not exceed self.max_graph_nodes
            max_nodes = min(max_nodes, self.max_graph_nodes)
        kg = KnowledgeGraph()

        # Handle wildcard query - get all nodes
        if node_label == "*":
            # First check total node count to determine if graph should be truncated
            count_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (n:base)
                    RETURN count(distinct n) AS total_nodes
                    $$) AS (total_nodes bigint)"""

            count_result = await self._query(count_query)
            total_nodes = count_result[0]["total_nodes"] if count_result else 0
            is_truncated = total_nodes > max_nodes

            # Get max_nodes with highest degrees
            query_nodes = f"""SELECT * FROM cypher('{self.graph_name}', $$
                    MATCH (n:base)
                    OPTIONAL MATCH (n)-[r]->()
                    RETURN id(n) as node_id, count(r) as degree
                $$) AS (node_id BIGINT, degree BIGINT)
                ORDER BY degree DESC
                LIMIT {max_nodes}"""
            node_results = await self._query(query_nodes)

            node_ids = [str(result["node_id"]) for result in node_results]

            logger.info(
                f"[{self.workspace}] Total nodes: {total_nodes}, Selected nodes: {len(node_ids)}"
            )

            if node_ids:
                formatted_ids = ", ".join(node_ids)
                # Construct batch query for subgraph within max_nodes
                query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                        WITH [{formatted_ids}] AS node_ids
                        MATCH (a)
                        WHERE id(a) IN node_ids
                        OPTIONAL MATCH (a)-[r]->(b)
                            WHERE id(b) IN node_ids
                        RETURN a, r, b
                    $$) AS (a AGTYPE, r AGTYPE, b AGTYPE)"""
                results = await self._query(query)

                # Process query results, deduplicate nodes and edges
                nodes_dict = {}
                edges_dict = {}
                for result in results:
                    # Process node a
                    if result.get("a") and isinstance(result["a"], dict):
                        node_a = result["a"]
                        node_id = str(node_a["id"])
                        if node_id not in nodes_dict and "properties" in node_a:
                            nodes_dict[node_id] = KnowledgeGraphNode(
                                id=node_id,
                                labels=[node_a["properties"]["entity_id"]],
                                properties=node_a["properties"],
                            )

                    # Process node b
                    if result.get("b") and isinstance(result["b"], dict):
                        node_b = result["b"]
                        node_id = str(node_b["id"])
                        if node_id not in nodes_dict and "properties" in node_b:
                            nodes_dict[node_id] = KnowledgeGraphNode(
                                id=node_id,
                                labels=[node_b["properties"]["entity_id"]],
                                properties=node_b["properties"],
                            )

                    # Process edge r
                    if result.get("r") and isinstance(result["r"], dict):
                        edge = result["r"]
                        edge_id = str(edge["id"])
                        if edge_id not in edges_dict:
                            edges_dict[edge_id] = KnowledgeGraphEdge(
                                id=edge_id,
                                type=edge["label"],
                                source=str(edge["start_id"]),
                                target=str(edge["end_id"]),
                                properties=edge["properties"],
                            )

                kg = KnowledgeGraph(
                    nodes=list(nodes_dict.values()),
                    edges=list(edges_dict.values()),
                    is_truncated=is_truncated,
                )
            else:
                # For single node query, use BFS algorithm
                kg = await self._bfs_subgraph(node_label, max_depth, max_nodes)

            logger.info(
                f"[{self.workspace}] Subgraph query successful | Node count: {len(kg.nodes)} | Edge count: {len(kg.edges)}"
            )
        else:
            # For non-wildcard queries, use the BFS algorithm
            kg = await self._bfs_subgraph(node_label, max_depth, max_nodes)
            logger.info(
                f"[{self.workspace}] Subgraph query for '{node_label}' successful | Node count: {len(kg.nodes)} | Edge count: {len(kg.edges)}"
            )

        return kg

    async def get_all_nodes(self) -> list[dict]:
        """Get all nodes in the graph.

        Returns:
            A list of all nodes, where each node is a dictionary of its properties
        """
        # Use native SQL to avoid Cypher wrapper overhead
        # Original: SELECT * FROM cypher(...) with MATCH (n:base)
        # Optimized: Direct table access for better performance
        query = f"""
            SELECT properties
            FROM {self.graph_name}.base
        """

        results = await self._query(query)
        nodes = []
        for result in results:
            if result.get("properties"):
                node_dict = result["properties"]

                # Process string result, parse it to JSON dictionary
                if isinstance(node_dict, str):
                    try:
                        node_dict = json.loads(node_dict)
                    except json.JSONDecodeError:
                        logger.warning(
                            f"[{self.workspace}] Failed to parse node string: {node_dict}"
                        )
                        continue

                # Add node id (entity_id) to the dictionary for easier access
                node_dict["id"] = node_dict.get("entity_id")
                nodes.append(node_dict)
        return nodes

    async def get_all_edges(self) -> list[dict]:
        """Get all edges in the graph.

        Returns:
            A list of all edges, where each edge is a dictionary of its properties
            (If 2 directional edges exist between the same pair of nodes, deduplication must be handled by the caller)
        """
        # Use native SQL to avoid Cartesian product (N×N) in Cypher MATCH
        # Original Cypher: MATCH (a:base)-[r]-(b:base) creates ~50 billion row combinations
        # Optimized: Start from edges table, join to nodes only to get entity_id
        # Performance: O(E) instead of O(N²), ~50,000x faster for large graphs
        query = f"""
            SELECT DISTINCT
                (ag_catalog.agtype_access_operator(VARIADIC ARRAY[a.properties, '"entity_id"'::agtype]))::text AS source,
                (ag_catalog.agtype_access_operator(VARIADIC ARRAY[b.properties, '"entity_id"'::agtype]))::text AS target,
                r.properties
            FROM {self.graph_name}."DIRECTED" r
            JOIN {self.graph_name}.base a ON r.start_id = a.id
            JOIN {self.graph_name}.base b ON r.end_id = b.id
        """

        results = await self._query(query)
        edges = []
        for result in results:
            edge_properties = result["properties"]

            # Process string result, parse it to JSON dictionary
            if isinstance(edge_properties, str):
                try:
                    edge_properties = json.loads(edge_properties)
                except json.JSONDecodeError:
                    logger.warning(
                        f"[{self.workspace}] Failed to parse edge properties string: {edge_properties}"
                    )
                    edge_properties = {}

            edge_properties["source"] = result["source"]
            edge_properties["target"] = result["target"]
            edges.append(edge_properties)
        return edges

    async def get_popular_labels(self, limit: int = 300) -> list[str]:
        """Get popular labels by node degree (most connected entities) using native SQL for performance."""
        try:
            # Native SQL query to calculate node degrees directly from AGE's underlying tables
            # This is significantly faster than using the cypher() function wrapper
            query = f"""
            WITH node_degrees AS (
                SELECT
                    node_id,
                    COUNT(*) AS degree
                FROM (
                    SELECT start_id AS node_id FROM {self.graph_name}._ag_label_edge
                    UNION ALL
                    SELECT end_id AS node_id FROM {self.graph_name}._ag_label_edge
                ) AS all_edges
                GROUP BY node_id
            )
            SELECT
                (ag_catalog.agtype_access_operator(VARIADIC ARRAY[v.properties, '"entity_id"'::agtype]))::text AS label
            FROM
                node_degrees d
            JOIN
                {self.graph_name}._ag_label_vertex v ON d.node_id = v.id
            WHERE
                ag_catalog.agtype_access_operator(VARIADIC ARRAY[v.properties, '"entity_id"'::agtype]) IS NOT NULL
            ORDER BY
                d.degree DESC,
                label ASC
            LIMIT $1;
            """
            results = await self._query(query, params={"limit": limit})
            labels = [
                result["label"] for result in results if result and "label" in result
            ]

            logger.debug(
                f"[{self.workspace}] Retrieved {len(labels)} popular labels (limit: {limit})"
            )
            return labels
        except Exception as e:
            logger.error(f"[{self.workspace}] Error getting popular labels: {str(e)}")
            return []

    async def search_labels(self, query: str, limit: int = 50) -> list[str]:
        """Search labels with fuzzy matching using native, parameterized SQL for performance and security."""
        query_lower = query.lower().strip()
        if not query_lower:
            return []

        try:
            # Re-implementing with the correct agtype access operator and full scoring logic.
            sql_query = f"""
            WITH ranked_labels AS (
                SELECT
                    (ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"entity_id"'::agtype]))::text AS label,
                    LOWER((ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"entity_id"'::agtype]))::text) AS label_lower
                FROM
                    {self.graph_name}._ag_label_vertex
                WHERE
                    ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"entity_id"'::agtype]) IS NOT NULL
                    AND LOWER((ag_catalog.agtype_access_operator(VARIADIC ARRAY[properties, '"entity_id"'::agtype]))::text) ILIKE $1
            )
            SELECT
                label
            FROM (
                SELECT
                    label,
                    CASE
                        WHEN label_lower = $2 THEN 1000
                        WHEN label_lower LIKE $3 THEN 500
                        ELSE (100 - LENGTH(label))
                    END +
                    CASE
                        WHEN label_lower LIKE $4 OR label_lower LIKE $5 THEN 50
                        ELSE 0
                    END AS score
                FROM
                    ranked_labels
            ) AS scored_labels
            ORDER BY
                score DESC,
                label ASC
            LIMIT $6;
            """
            params = (
                f"%{query_lower}%",  # For the main ILIKE clause ($1)
                query_lower,  # For exact match ($2)
                f"{query_lower}%",  # For prefix match ($3)
                f"% {query_lower}%",  # For word boundary (space) ($4)
                f"%_{query_lower}%",  # For word boundary (underscore) ($5)
                limit,  # For LIMIT ($6)
            )
            results = await self._query(sql_query, params=dict(enumerate(params, 1)))
            labels = [
                result["label"] for result in results if result and "label" in result
            ]

            logger.debug(
                f"[{self.workspace}] Search query '{query}' returned {len(labels)} results (limit: {limit})"
            )
            return labels
        except Exception as e:
            logger.error(
                f"[{self.workspace}] Error searching labels with query '{query}': {str(e)}"
            )
            return []

    async def drop(self) -> dict[str, str]:
        """Drop the storage"""
        try:
            drop_query = f"""SELECT * FROM cypher('{self.graph_name}', $$
                            MATCH (n)
                            DETACH DELETE n
                            $$) AS (result agtype)"""

            await self._query(drop_query, readonly=False)
            return {
                "status": "success",
                "message": f"workspace '{self.workspace}' graph data dropped",
            }
        except Exception as e:
            logger.error(f"[{self.workspace}] Error dropping graph: {e}")
            return {"status": "error", "message": str(e)}


# Note: Order matters! More specific namespaces (e.g., "full_entities") must come before
# more general ones (e.g., "entities") because is_namespace() uses endswith() matching
