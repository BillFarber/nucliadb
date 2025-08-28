# Copyright (C) 2021 Bosutech XXI S.L.
#
# nucliadb is offered under the AGPL v3.0 and as commercial software.
# For commercial licensing, contact us at info@nuclia.com.
#
# AGPL:
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import asyncio
import logging
from copy import deepcopy
from typing import Any, Iterator, Optional
from uuid import uuid4

import backoff
from cachetools import TTLCache
from marklogic import Client
from marklogic.documents import Document
from nidx_protos.nodereader_pb2 import FilterExpression, SearchRequest
from nidx_protos.noderesources_pb2 import (
    IndexParagraph,
    Resource,
    VectorSentence,
)
from pydantic import BaseModel

from nucliadb.common.counters import IndexCounts
from nucliadb.common.external_index_providers.base import (
    ExternalIndexManager,
    ExternalIndexProviderType,
    QueryResults,
    TextBlockMatch,
    VectorsetExternalIndex,
)
from nucliadb.common.external_index_providers.exceptions import (
    ExternalIndexCreationError,
)
from nucliadb.common.ids import ParagraphId, VectorId
from nucliadb_models.search import SCORE_TYPE, TextPosition
from nucliadb_protos import knowledgebox_pb2 as kb_pb2
from nucliadb_protos import utils_pb2
from nucliadb_telemetry.metrics import Observer
from nucliadb_utils.utilities import get_endecryptor, get_marklogic_client

logger = logging.getLogger(__name__)

manager_observer = Observer("marklogic_index_manager", labels={"operation": ""})


# Labels that should be excluded from MarkLogic metadata to avoid size limits
DISCARDED_LABEL_PREFIXES = [
    # NER-related labels are not supported in the MarkLogic integration
    # to avoid potential metadata size issues
    "/e/",
    # Processing status labels are only needed for the catalog endpoint
    "/n/s",
]

# Cache for index counts to avoid frequent API calls
COUNTERS_CACHE = TTLCache(maxsize=1024, ttl=60)  # type: ignore


class MarkLogicQueryResults(QueryResults):
    type: ExternalIndexProviderType = ExternalIndexProviderType.MARKLOGIC
    results: dict[str, Any]  # MarkLogic response structure

    def iter_matching_text_blocks(self) -> Iterator[TextBlockMatch]:
        """
        Iterates over MarkLogic search results and converts them to
        TextBlockMatch objects.
        """
        matches = self.results.get("matches", [])
        for order, match in enumerate(matches):
            try:
                vector_id = VectorId.from_string(match.get("id", ""))
                paragraph_id = ParagraphId.from_vector_id(vector_id)
            except ValueError:  # pragma: no cover
                logger.error(f"Invalid MarkLogic vector id: {match.get('id')}")
                continue

            metadata = match.get("metadata", {})
            vector_metadata = VectorMetadata.model_validate(metadata)

            yield TextBlockMatch(
                paragraph_id=paragraph_id,
                text=None,  # To be filled by the results hydrator
                score=match.get("score", 0.0),
                score_type=SCORE_TYPE.VECTOR,
                order=order,
                fuzzy_search=False,  # semantic search doesn't use fuzziness
                is_a_table=vector_metadata.is_a_table or False,
                page_with_visual=vector_metadata.page_with_visual or False,
                representation_file=vector_metadata.representation_file,
                paragraph_labels=vector_metadata.paragraph_labels or [],
                field_labels=vector_metadata.field_labels or [],
                position=TextPosition(
                    page_number=vector_metadata.page_number,
                    index=vector_id.index or 0,
                    start=paragraph_id.paragraph_start,
                    end=paragraph_id.paragraph_end,
                    start_seconds=list(
                        map(int, vector_metadata.position_start_seconds or [])
                    ),
                    end_seconds=list(
                        map(int, vector_metadata.position_end_seconds or [])
                    ),
                ),
            )


class IndexNotFound(Exception):
    pass


class VectorMetadata(BaseModel):
    """
    Models the metadata stored with each vector in MarkLogic.
    """

    # Resource and field identification
    rid: str
    field_type: str
    field_id: str

    # Date range filtering
    date_created: Optional[int] = None
    date_modified: Optional[int] = None

    # Label filtering
    paragraph_labels: Optional[list[str]] = None
    field_labels: Optional[list[str]] = None

    # Security
    security_public: bool = True
    security_ids_with_access: Optional[list[str]] = None

    # Position metadata
    position_start_seconds: Optional[list[str]] = None
    position_end_seconds: Optional[list[str]] = None
    page_number: Optional[int] = None

    # AI-tables metadata
    page_with_visual: Optional[bool] = None
    is_a_table: Optional[bool] = None
    representation_file: Optional[str] = None


class MarkLogicIndexManager(ExternalIndexManager):
    type = ExternalIndexProviderType.MARKLOGIC
    supports_rollover = True

    def __init__(
        self,
        kbid: str,
        host: str,
        port: int,
        username: str,
        password: str,
        database: str,
        indexes: dict[str, kb_pb2.MarkLogicIndexMetadata],
        upsert_parallelism: int = 3,
        delete_parallelism: int = 2,
        upsert_timeout: float = 10.0,
        delete_timeout: float = 10.0,
        query_timeout: float = 10.0,
        default_vectorset: Optional[str] = None,
        rollover_indexes: Optional[
            dict[str, kb_pb2.MarkLogicIndexMetadata]
        ] = None,
    ):
        super().__init__(kbid=kbid)
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.database = database
        self.indexes = indexes
        self.rollover_indexes = rollover_indexes or {}
        self.upsert_parallelism = upsert_parallelism
        self.delete_parallelism = delete_parallelism
        self.upsert_timeout = upsert_timeout
        self.delete_timeout = delete_timeout
        self.query_timeout = query_timeout
        self.default_vectorset = default_vectorset

        marklogic_config: kb_pb2.CreateMarkLogicConfig = (
            kb_pb2.CreateMarkLogicConfig(
                host=self.host,
                port=self.port,
                username=self.username,
                password=self.password,
                database=self.database,
            )
        )
        self.marklogic_client: Client = get_marklogic_client(marklogic_config)

    @classmethod
    async def create_indexes(
        cls,
        kbid: str,
        request: kb_pb2.CreateExternalIndexProviderMetadata,
        indexes: list[VectorsetExternalIndex],
    ) -> kb_pb2.StoredExternalIndexProviderMetadata:
        """
        Creates MarkLogic indexes for the given vectorsets.
        """
        created_indexes = []
        metadata = kb_pb2.StoredExternalIndexProviderMetadata(
            type=kb_pb2.ExternalIndexProviderType.MARKLOGIC
        )
        marklogic_client: Client = get_marklogic_client(
            request.marklogic_config
        )

        # Extract configuration from request
        metadata.marklogic_config.host = request.marklogic_config.host
        metadata.marklogic_config.port = request.marklogic_config.port
        metadata.marklogic_config.username = request.marklogic_config.username
        metadata.marklogic_config.encrypted_password = (
            get_endecryptor().encrypt(request.marklogic_config.password)
        )
        metadata.marklogic_config.database = request.marklogic_config.database

        try:
            for index in indexes:
                vectorset_id = index.vectorset_id
                index_name = cls.get_index_name(kbid, vectorset_id)
                index_dimension = index.dimension
                similarity_metric = to_marklogic_similarity_function(
                    index.similarity
                )

                logger.info(
                    "Creating MarkLogic index",
                    extra={
                        "kbid": kbid,
                        "index_name": index_name,
                        "similarity": similarity_metric,
                        "vector_dimension": index_dimension,
                        "vectorset_id": vectorset_id,
                        "database": request.marklogic_config.database,
                    },
                )

                index_uri = await cls._create_marklogic_index(
                    marklogic_client,
                    index_name=index_name,
                    dimension=index_dimension,
                    similarity=similarity_metric,
                )

                created_indexes.append(index_name)
                metadata.marklogic_config.indexes[vectorset_id].CopyFrom(
                    kb_pb2.MarkLogicIndexMetadata(
                        index_name=index_name,
                        index_uri=index_uri,
                        vector_dimension=index.dimension,
                        similarity=index.similarity,
                    )
                )
        except Exception as exc:
            # Rollback created indexes
            for index_name in created_indexes:
                try:
                    await cls._delete_marklogic_index(
                        marklogic_client,
                        index_name=index_name,
                    )
                except Exception:
                    logger.exception(
                        "Could not rollback created MarkLogic indexes"
                    )
            raise ExternalIndexCreationError("marklogic", str(exc)) from exc

        return metadata

    @classmethod
    async def delete_indexes(
        cls,
        kbid: str,
        stored: kb_pb2.StoredExternalIndexProviderMetadata,
    ) -> None:
        """
        Deletes all MarkLogic indexes for the knowledge box.
        """
        marklogic_config: kb_pb2.CreateMarkLogicConfig = (
            kb_pb2.CreateMarkLogicConfig(
                host=stored.marklogic_config.host,
                port=stored.marklogic_config.port,
                username=stored.marklogic_config.username,
                password=get_endecryptor().decrypt(
                    stored.marklogic_config.encrypted_password
                ),
                database=stored.marklogic_config.database,
            )
        )
        marklogic_client: Client = get_marklogic_client(marklogic_config)

        for index_metadata in stored.marklogic_config.indexes.values():
            index_name = index_metadata.index_name
            try:
                logger.info(
                    "Deleting MarkLogic index",
                    extra={"kbid": kbid, "index_name": index_name},
                )
                await cls._delete_marklogic_index(
                    marklogic_client,
                    index_name=index_name,
                )
            except Exception:
                logger.exception(
                    "Error deleting MarkLogic index",
                    extra={"kbid": kbid, "index_name": index_name},
                )

    @classmethod
    async def _create_marklogic_index(
        cls,
        marklogic_client: Client,
        index_name: str,
        dimension: int,
        similarity: str,
    ) -> str:
        """
        Creates a vector index in MarkLogic.
        Returns the index URI.
        """
        logger.info(
            f"Creating MarkLogic vector index: {index_name} with dimension {dimension}"
        )
        template = await cls._build_template(index_name, dimension)
        json_doc = Document(
            f"/{index_name}.json",
            template,
            collections=["TDE", "http://marklogic.com/xdmp/tde"],
        )
        marklogic_client.documents.write(
            json_doc, params={"database": "Schemas"}
        )

        return f"{index_name}"

    @classmethod
    async def _build_template(
        cls, index_name: str, dimension: int
    ) -> dict[str, Any]:
        """
        Builds the template for the MarkLogic vector index.
        """
        return {
            "template": {
                "context": "/",
                "collections": [index_name],
                "rows": [
                    {
                        "schemaName": "nucliaVectors",
                        "viewName": index_name,
                        "columns": [
                            {
                                "name": "id",
                                "scalarType": "string",
                                "val": "id",
                            },
                            {
                                "name": "vector",
                                "scalarType": "vector",
                                "val": "vec:vector(vector)",
                                "dimension": str(dimension),
                                "invalidValues": "reject",
                            },
                            {
                                "name": "rid",
                                "scalarType": "string",
                                "val": "metadata/rid",
                            },
                            {
                                "name": "page_number",
                                "scalarType": "int",
                                "val": "metadata/page_number",
                            },
                        ],
                    }
                ],
            }
        }

    @classmethod
    @backoff.on_exception(
        backoff.expo,
        (Exception,),  # MarkLogicAPIError would go here
        jitter=backoff.random_jitter,
        max_tries=3,
    )
    async def _delete_marklogic_index(
        cls,
        marklogic_client: Client,
        index_name: str,
    ) -> None:
        """
        Deletes a vector index from MarkLogic.
        """
        logger.info(f"Deleting MarkLogic vector index: {index_name}")
        index_uri = f"/{index_name}.json"
        delete_script = f'declareUpdate(); xdmp.documentDelete("{index_uri}")'
        response = marklogic_client.eval(
            javascript=delete_script,
            params={"database": "Schemas"},
        )
        print(response)

    async def rollover_create_indexes(
        self, stored: kb_pb2.StoredExternalIndexProviderMetadata
    ) -> kb_pb2.StoredExternalIndexProviderMetadata:
        """
        Creates new indexes for rollover process.
        """
        result = kb_pb2.StoredExternalIndexProviderMetadata()
        result.CopyFrom(stored)

        marklogic_config: kb_pb2.CreateMarkLogicConfig = (
            kb_pb2.CreateMarkLogicConfig(
                host=stored.marklogic_config.host,
                port=stored.marklogic_config.port,
                username=stored.marklogic_config.username,
                password=get_endecryptor().decrypt(
                    stored.marklogic_config.password
                ),
                database=stored.marklogic_config.database,
            )
        )
        marklogic_client: Client = get_marklogic_client(marklogic_config)

        created_indexes = []
        try:
            for vectorset_id, index in stored.marklogic_config.indexes.items():
                rollover_index_name = self.get_index_name(self.kbid, vectorset_id)
                index_dimension = index.vector_dimension
                similarity_metric = to_marklogic_similarity_function(
                    index.similarity
                )

                logger.info(
                    "Creating MarkLogic rollover index",
                    extra={
                        "kbid": self.kbid,
                        "index_name": index.index_name,
                        "rollover_index_name": rollover_index_name,
                        "similarity": similarity_metric,
                        "vector_dimension": index_dimension,
                        "vectorset_id": vectorset_id,
                    },
                )

                try:
                    index_uri = await self._create_marklogic_index(
                        marklogic_client,
                        index_name=rollover_index_name,
                        dimension=index_dimension,
                        similarity=similarity_metric,
                    )

                    result.marklogic_config.indexes[vectorset_id].MergeFrom(
                        kb_pb2.MarkLogicIndexMetadata(
                            index_name=rollover_index_name,
                            index_uri=index_uri,
                            vector_dimension=index_dimension,
                            similarity=index.similarity,
                        )
                    )
                    created_indexes.append(rollover_index_name)
                except Exception as exc:
                    raise ExternalIndexCreationError(
                        "marklogic", str(exc)
                    ) from exc
        except Exception:
            # Rollback any created indexes
            for index_name in created_indexes:
                try:
                    await self._delete_marklogic_index(
                        marklogic_client,
                        index_name=index_name,
                    )
                except Exception:
                    logger.exception(
                        "Could not rollback created MarkLogic index",
                        extra={
                            "kbid": self.kbid,
                            "index_name": index_name,
                        },
                    )
            raise

        # Update rollover indexes
        self.rollover_indexes.clear()
        self.rollover_indexes = dict(result.marklogic_config.indexes)
        return result

    async def rollover_cutover_indexes(self) -> None:
        """
        Cutover to rollover indexes by deleting old indexes.
        """
        assert (
            len(self.rollover_indexes) > 0
        ), "No rollover indexes to cutover to"

        marklogic_config: kb_pb2.CreateMarkLogicConfig = (
            kb_pb2.CreateMarkLogicConfig(
                host=self.stored_config.marklogic_config.host,
                port=self.stored_config.marklogic_config.port,
                username=self.stored_config.marklogic_config.username,
                password=get_endecryptor().decrypt(
                    self.stored_config.marklogic_config.password
                ),
                database=self.stored_config.marklogic_config.database,
            )
        )
        marklogic_client: Client = get_marklogic_client(marklogic_config)

        for index in self.indexes.values():
            index_name = index.index_name
            try:
                await self._delete_marklogic_index(
                    marklogic_client,
                    index_name=index_name,
                )
            except Exception:
                logger.exception(
                    "Error deleting MarkLogic index on cutover",
                    extra={"kbid": self.kbid, "index_name": index_name},
                )

        self.indexes.clear()
        self.indexes.update(self.rollover_indexes)

    @classmethod
    def get_index_name(cls, kbid, vectorset_id) -> str:
        """
        Generates a unique index name for MarkLogic.
        MarkLogic index names should be valid NCNames (XML names).
        """
        return f"nuclia_{kbid}_{vectorset_id}".replace("-", "_")

    async def _delete_resource_from_index(
        self, index_uri: str, resource_uuid: str
    ) -> None:
        """
        Deletes all vectors for a resource from a specific MarkLogic index.
        """
        with manager_observer({"operation": "delete_by_resource_prefix"}):
            resource_collection = (
                f"NucliaDB_resource_{resource_uuid}_index_{index_uri}"
            )
            logger.info(
                f"Deleting resource {resource_collection} from MarkLogic index {index_uri}"
            )
            delete_script = f'declareUpdate(); xdmp.collectionDelete("{resource_collection}")'
            print(delete_script)
            response = self.marklogic_client.eval(
                javascript=delete_script,
            )
            print(response)

    async def _delete_resource(self, resource_uuid: str) -> None:
        """
        Deletes a resource from all MarkLogic indexes.
        """
        delete_tasks = []
        for index in self.indexes.values():
            index_uri = index.index_uri
            delete_tasks.append(
                asyncio.create_task(
                    self._delete_resource_from_index(
                        index_uri=index_uri,
                        resource_uuid=resource_uuid,
                    )
                )
            )
        if len(delete_tasks) > 0:
            await asyncio.gather(*delete_tasks)

    def get_index_uri(self, vectorset_id: str, rollover: bool = False) -> str:
        """
        Gets the index URI for a vectorset.
        """
        if rollover:
            return self.rollover_indexes[vectorset_id].index_uri
        else:
            return self.indexes[vectorset_id].index_uri

    def get_prefixes_to_delete(
        self, index_data: Resource
    ) -> dict[str, set[str]]:
        """
        Extracts prefixes that need to be deleted from the index data.
        """
        return {
            vectorset_id: set(prefixes_list.items)
            for vectorset_id, prefixes_list in index_data.vector_prefixes_to_delete.items()
        }

    async def _index_resource(
        self,
        resource_uuid: str,
        index_data: Resource,
        to_rollover_indexes: bool = False,
    ) -> None:
        """
        Indexes a NucliaDB resource into MarkLogic.
        """
        # First, delete any existing vectors for the resource prefixes
        delete_tasks = []
        for vectorset, prefixes_to_delete in self.get_prefixes_to_delete(
            index_data
        ).items():
            index_uri = self.get_index_uri(
                vectorset_id=vectorset, rollover=to_rollover_indexes
            )
            delete_tasks.append(
                asyncio.create_task(
                    self._delete_by_prefix_from_index(
                        index_uri=index_uri,
                        prefixes_to_delete=prefixes_to_delete,
                    )
                )
            )
        if len(delete_tasks) > 0:
            await asyncio.gather(*delete_tasks)

        # Compute base metadata for vectors
        with manager_observer({"operation": "compute_base_vector_metadatas"}):
            base_vector_metadatas: dict[str, VectorMetadata] = (
                await self.compute_base_vector_metadatas(
                    index_data, resource_uuid
                )
            )

        # Compute vectors for each vectorset
        with manager_observer({"operation": "compute_vectorset_vectors"}):
            vectorset_vectors: dict[str, list[dict[str, Any]]] = (
                await self.compute_vectorset_vectors(
                    index_data, base_vector_metadatas
                )
            )

        # Upsert vectors to MarkLogic
        upsert_tasks = []
        for vectorset_id, vectors in vectorset_vectors.items():
            index_uri = self.get_index_uri(
                vectorset_id=vectorset_id, rollover=to_rollover_indexes
            )
            resource_collection = (
                f"NucliaDB_resource_{resource_uuid}_index_{index_uri}"
            )
            upsert_tasks.append(
                asyncio.create_task(
                    self._upsert_to_index(
                        index_uri=index_uri,
                        resource_collection=resource_collection,
                        vectors=vectors,
                    )
                )
            )
        if len(upsert_tasks) > 0:
            await asyncio.gather(*upsert_tasks)

    async def _upsert_to_index(
        self,
        index_uri: str,
        resource_collection: str,
        vectors: list[dict[str, Any]],
    ) -> None:
        """
        Upserts vectors to a MarkLogic index.
        """
        if len(vectors) == 0:  # pragma: no cover
            return

        with manager_observer({"operation": "upsert_vectors"}):
            logger.info(
                f"Upserting {len(vectors)} vectors to MarkLogic index {index_uri}"
            )
            for vector in vectors:
                json_doc = Document(
                    f"{index_uri}/{vector['id']}.json",
                    vector,
                    collections=["NucliaDB", resource_collection, index_uri],
                )
                self.marklogic_client.documents.write(json_doc)

    async def _delete_by_prefix_from_index(
        self, index_uri: str, prefixes_to_delete: set[str]
    ) -> None:
        """
        Deletes vectors by prefix from a MarkLogic index.
        """
        if len(prefixes_to_delete) == 0:  # pragma: no cover
            return

        with manager_observer({"operation": "delete_by_prefix"}):
            for prefix in prefixes_to_delete:
                # TODO: Implement MarkLogic vector deletion by prefix
                logger.info(
                    f"Deleting vectors with prefix {prefix} from MarkLogic index {index_uri}"
                )

    async def compute_base_vector_metadatas(
        self, index_data: Resource, resource_uuid: str
    ) -> dict[str, VectorMetadata]:
        """
        Computes base metadata for vectors that is common across all vectorsets.
        """
        return await asyncio.to_thread(
            self._compute_base_vector_metadatas, index_data, resource_uuid
        )

    def _compute_base_vector_metadatas(
        self, index_data: Resource, resource_uuid: str
    ) -> dict[str, VectorMetadata]:
        """
        Synchronous computation of base vector metadata.
        """
        metadatas: dict[str, VectorMetadata] = {}
        security_public = True
        security_ids_with_access = None

        if index_data.HasField("security"):
            security_public = False
            security_ids_with_access = list(
                set(index_data.security.access_groups)
            )

        resource_labels = set(index_data.labels)
        date_created = index_data.metadata.created.ToSeconds()
        date_modified = index_data.metadata.modified.ToSeconds()

        for field_id, text_info in index_data.texts.items():
            field_labels = set(text_info.labels)
            field_paragraphs = index_data.paragraphs.get(field_id)
            if field_paragraphs is None:
                logger.info(
                    "Paragraphs not found for field",
                    extra={
                        "kbid": self.kbid,
                        "rid": resource_uuid,
                        "field_id": field_id,
                    },
                )
                continue

            paragraph: IndexParagraph
            for paragraph_id, paragraph in field_paragraphs.paragraphs.items():
                fid = ParagraphId.from_string(paragraph_id).field_id
                vector_metadata = VectorMetadata(
                    rid=resource_uuid,
                    field_type=fid.type,
                    field_id=fid.key,
                    date_created=date_created,
                    date_modified=date_modified,
                    security_public=security_public,
                    security_ids_with_access=security_ids_with_access,
                )
                metadatas[paragraph_id] = vector_metadata

                final_field_labels = resource_labels.union(field_labels)
                if final_field_labels:
                    vector_metadata.field_labels = unique(
                        discard_labels(list(final_field_labels))
                    )

                final_paragraph_labels = paragraph.labels
                if final_paragraph_labels:
                    vector_metadata.paragraph_labels = unique(
                        discard_labels(list(final_paragraph_labels))
                    )
        return metadatas

    async def compute_vectorset_vectors(
        self,
        index_data: Resource,
        base_vector_metadatas: dict[str, VectorMetadata],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Computes vectors for each vectorset.
        """
        return await asyncio.to_thread(
            self._compute_vectorset_vectors, index_data, base_vector_metadatas
        )

    def _compute_vectorset_vectors(
        self,
        index_data: Resource,
        base_vector_metadatas: dict[str, VectorMetadata],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Synchronous computation of vectorset vectors.
        """
        vectorset_vectors: dict[str, list[dict[str, Any]]] = {}

        for index_paragraph_id, index_paragraph in iter_paragraphs(index_data):
            vectorset_iterators = {}
            if index_paragraph.sentences and self.default_vectorset:
                vectorset_iterators[self.default_vectorset] = (
                    index_paragraph.sentences.items()
                )
            for (
                vectorset_id,
                vector_sentences,
            ) in index_paragraph.vectorsets_sentences.items():
                if vector_sentences.sentences:
                    vectorset_iterators[vectorset_id] = (
                        vector_sentences.sentences.items()
                    )

            vector_sentence: VectorSentence
            for vectorset_id, sentences_iterator in vectorset_iterators.items():
                for sentence_id, vector_sentence in sentences_iterator:
                    vector_metadata_to_copy = base_vector_metadatas.get(
                        index_paragraph_id
                    )
                    if vector_metadata_to_copy is None:
                        logger.warning(
                            f"Metadata not found for sentences of paragraph {index_paragraph_id}"
                        )
                        continue

                    # Copy the initial metadata
                    vector_metadata = deepcopy(vector_metadata_to_copy)

                    # AI-tables metadata
                    if vector_sentence.metadata.page_with_visual:
                        vector_metadata.page_with_visual = True
                    if vector_sentence.metadata.representation.is_a_table:
                        vector_metadata.is_a_table = True
                    if vector_sentence.metadata.representation.file:
                        vector_metadata.representation_file = (
                            vector_sentence.metadata.representation.file
                        )

                    # Video positions
                    if len(vector_sentence.metadata.position.start_seconds):
                        vector_metadata.position_start_seconds = list(
                            map(
                                str,
                                vector_sentence.metadata.position.start_seconds,
                            )
                        )
                    if len(vector_sentence.metadata.position.end_seconds):
                        vector_metadata.position_end_seconds = list(
                            map(
                                str,
                                vector_sentence.metadata.position.end_seconds,
                            )
                        )
                    vector_metadata.page_number = (
                        vector_sentence.metadata.position.page_number
                    )

                    marklogic_vector = {
                        "id": sentence_id,
                        "vector": list(vector_sentence.vector),
                        "metadata": vector_metadata.model_dump(
                            exclude_none=True
                        ),
                    }

                    vectors = vectorset_vectors.setdefault(vectorset_id, [])
                    vectors.append(marklogic_vector)

        return vectorset_vectors

    async def _query(self, request: SearchRequest) -> MarkLogicQueryResults:
        """
        Queries MarkLogic index for similar vectors.
        """
        if len(request.vector) == 0:
            return MarkLogicQueryResults(results={"matches": []})

        vectorset_id = (
            request.vectorset or self.default_vectorset or "__default__"
        )
        index_uri = self.get_index_uri(vectorset_id=vectorset_id)
        index_name = self.get_index_name(self.kbid, vectorset_id)

        # Convert search request to MarkLogic query
        marklogic_filter = convert_to_marklogic_filter(request)
        top_k = request.result_per_page

        query_results = await self._execute_marklogic_query(
            index_uri=index_uri,
            index_name=index_name,
            vector=list(request.vector),
            top_k=top_k,
            filter_conditions=marklogic_filter,
            min_score=request.min_score_semantic,
        )

        return MarkLogicQueryResults(results=query_results)

    async def _execute_marklogic_query(
        self,
        index_uri: str,
        index_name: str,
        vector: list[float],
        top_k: int,
        filter_conditions: Optional[dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """
        Executes a vector similarity query against MarkLogic.
        """
        logger.info(
            f"Executing MarkLogic query on index {index_name} with top_k={top_k}"
        )

        optic_vector_query = self._build_optic_vector_query(
            query_embedding=vector,
            tde_schema="nucliaVectors",
            tde_view=index_name,
        )
        logger.info(
            f"optic_vector_query:\n {optic_vector_query}"
        )
        response = self.marklogic_client.eval(
            javascript=optic_vector_query,
            params={"database": self.database}
        )
        logger.info(
            f"response:\n {response}"
        )

        matches = []  # type: ignore[assignment]
        if response:
            for match in response:
                matches.append({
                    "id": match.get("id"),
                    "score": match.get("score"),
                    "values": match.get("values"),
                "metadata": {
                    "rid": match.get("rid"),
                    "page_number": match.get("page_number"),
                    "field_type": "",
                    "field_id": "",
                }
            })

        # Placeholder return structure
        return {
            "matches": matches,
            "query_metadata": {
                "index_uri": index_uri,
                "vector_dimension": len(vector),
                "top_k": top_k,
                "min_score": min_score,
            },
        }

    def _build_optic_vector_query(self, query_embedding, tde_schema, tde_view):
        max_results: int = 10
        return """
            const op = require('/MarkLogic/optic');
            const ovec = require('/MarkLogic/optic/optic-vec.xqy');
            const qv = vec.vector({})

            const rows = op.fromView('{}','{}','')
              .bind([
                op.as('values', op.col('vector'))
            ])
            .bind(op.as('score', op.vec.cosine(op.vec.vector(op.col('vector')),qv)))
            .select(['id','values','score','rid','page_number'])
            .orderBy(op.desc(op.col('score')))
            .limit({})
            .result();
            rows;
        """.format(
            query_embedding,
            tde_schema,
            tde_view,
            max_results
        )


    async def _get_index_counts(self) -> IndexCounts:
        """
        Gets index statistics from MarkLogic.
        """
        if self.kbid in COUNTERS_CACHE:
            return COUNTERS_CACHE[self.kbid]

        total = IndexCounts(fields=0, paragraphs=0, sentences=0, size_bytes=0)
        tasks = []
        vectorset_results: dict[str, IndexCounts] = {}

        for vectorset_id in self.indexes.keys():
            tasks.append(
                asyncio.create_task(
                    self._get_vectorset_index_counts(
                        vectorset_id, vectorset_results
                    )
                )
            )
        if len(tasks) > 0:
            await asyncio.gather(*tasks)

        for _, counts in vectorset_results.items():
            total.paragraphs += counts.paragraphs
            total.sentences += counts.sentences

        COUNTERS_CACHE[self.kbid] = total
        return total

    async def _get_vectorset_index_counts(
        self, vectorset_id: str, results: dict[str, IndexCounts]
    ) -> None:
        """
        Gets statistics for a specific vectorset index.
        """
        index_uri = self.get_index_uri(vectorset_id=vectorset_id)
        try:
            # TODO: Implement MarkLogic index statistics retrieval
            index_stats = await self._get_marklogic_index_stats(index_uri)
            results[vectorset_id] = IndexCounts(
                fields=0,
                paragraphs=index_stats.get("total_vectors", 0),
                sentences=index_stats.get("total_vectors", 0),
                size_bytes=index_stats.get("index_size_bytes", 0),
            )
        except Exception:
            logger.exception(
                "Error getting MarkLogic index stats",
                extra={
                    "kbid": self.kbid,
                    "provider": self.type.value,
                    "index_uri": index_uri,
                },
            )

    async def _get_marklogic_index_stats(
        self, index_uri: str
    ) -> dict[str, Any]:
        """
        Gets statistics for a MarkLogic index.
        """
        # TODO: Implement MarkLogic index statistics API call
        logger.info(f"Getting MarkLogic index stats for {index_uri}")
        return {
            "total_vectors": 0,
            "index_size_bytes": 0,
        }


def discard_labels(labels: list[str]) -> list[str]:
    """
    Filters out labels that should not be included in MarkLogic metadata.
    """
    return [
        label
        for label in labels
        if not any(
            label.startswith(prefix) for prefix in DISCARDED_LABEL_PREFIXES
        )
    ]


def unique(labels: list[str]) -> list[str]:
    """
    Returns unique labels preserving order.
    """
    return list(dict.fromkeys(labels))


def convert_to_marklogic_filter(
    request: SearchRequest,
) -> Optional[dict[str, Any]]:
    """
    Converts NucliaDB search request filters to MarkLogic query conditions.
    """
    conditions = []

    if request.HasField("field_filter"):
        conditions.append(
            convert_filter_expression("field_labels", request.field_filter)
        )
    if request.HasField("paragraph_filter"):
        conditions.append(
            convert_filter_expression(
                "paragraph_labels", request.paragraph_filter
            )
        )

    if len(request.security.access_groups):
        # Security filtering
        security_condition = {
            "$or": [
                {"security_public": True},
                {
                    "security_ids_with_access": {
                        "$in": list(set(request.security.access_groups))
                    }
                },
            ]
        }
        conditions.append(security_condition)

    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def convert_filter_expression(
    field: str, expression: FilterExpression, negative: bool = False
) -> dict[str, Any]:
    """
    Converts internal filter expressions to MarkLogic query conditions.
    """
    kind = expression.WhichOneof("expr")

    if kind == "bool_and":
        operator = "$or" if negative else "$and"
        return {
            operator: [
                convert_filter_expression(
                    field, sub_expression, negative=negative
                )
                for sub_expression in expression.bool_and.operands
            ]
        }
    elif kind == "bool_or":
        operator = "$and" if negative else "$or"
        return {
            operator: [
                convert_filter_expression(
                    field, sub_expression, negative=negative
                )
                for sub_expression in expression.bool_or.operands
            ]
        }
    elif kind == "bool_not":
        return convert_filter_expression(
            field, expression.bool_not, negative=not negative
        )
    elif kind == "resource":
        operator = "$ne" if negative else "$eq"
        return {"rid": {operator: expression.resource.resource_id}}
    elif kind == "field":
        field_id = expression.field.field_type
        if expression.field.HasField("field_id"):
            field_id += f"/{expression.field.field_id}"
        operator = "$ne" if negative else "$eq"
        return {"field_id": {operator: field_id}}
    elif kind == "keyword":
        raise ValueError("Cannot filter by keywords in MarkLogic vector search")
    elif kind == "date":
        date_field = (
            "date_created"
            if expression.date.field
            == FilterExpression.DateRangeFilter.DateField.CREATED
            else "date_modified"
        )
        if negative:
            terms = []
            if expression.date.HasField("since"):
                terms.append(
                    {date_field: {"$lt": expression.date.since.ToSeconds()}}
                )
            if expression.date.HasField("until"):
                terms.append(
                    {date_field: {"$gt": expression.date.until.ToSeconds()}}
                )

            if len(terms) == 2:
                return {"$or": terms}
            elif len(terms) == 1:
                return terms[0]
            else:
                raise ValueError(f"Invalid filter expression: {expression}")
        else:
            terms = []
            if expression.date.HasField("since"):
                terms.append(
                    {date_field: {"$gte": expression.date.since.ToSeconds()}}
                )
            if expression.date.HasField("until"):
                terms.append(
                    {date_field: {"$lte": expression.date.until.ToSeconds()}}
                )

            if len(terms) == 2:
                return {"$and": terms}
            elif len(terms) == 1:
                return terms[0]
            else:
                raise ValueError(f"Invalid filter expression: {expression}")
    elif kind == "facet":
        operator = "$nin" if negative else "$in"
        return {field: {operator: [expression.facet.facet]}}
    else:
        raise ValueError(f"Invalid filter expression: {expression}")


def iter_paragraphs(resource: Resource) -> Iterator[tuple[str, IndexParagraph]]:
    """
    Iterates over all paragraphs in a resource.
    """
    for _, paragraphs in resource.paragraphs.items():
        for paragraph_id, paragraph in paragraphs.paragraphs.items():
            yield paragraph_id, paragraph


def to_marklogic_similarity_function(
    similarity: utils_pb2.VectorSimilarity.ValueType,
) -> str:
    """
    Converts NucliaDB similarity function to MarkLogic similarity function.
    """
    return {
        utils_pb2.VectorSimilarity.COSINE: "cosine",
        utils_pb2.VectorSimilarity.DOT: "dot-product",
    }[similarity]
