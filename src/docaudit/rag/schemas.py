"""The data that flows through retrieval and answering.

Two ideas shape these models:

1. A ``Chunk`` remembers where it came from (document, section). Without that, a citation is
   impossible and an auditor can't check anything.
2. ``ClauseAnswer`` is what the model must reply with, so its ``description`` fields are prompt
   text, just like in ``extraction/schemas.py``.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from docaudit.llm.types import Usage


class SourceDocument(BaseModel):
    """A contract or policy, as plain text. See ``chunking.py`` for the expected format."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str
    text: str


class Chunk(BaseModel):
    """One retrievable piece of a document."""

    model_config = ConfigDict(frozen=True)

    id: str
    doc_id: str
    doc_title: str
    section: str
    heading: str = ""
    text: str

    @property
    def search_text(self) -> str:
        """What gets embedded and indexed: the clause plus the context it can't live without.

        Every contract has an almost identical clause 4.1. Only the title says whose it is.
        """
        return f"{self.doc_title}\n{self.heading}\n{self.text}"


class SearchHit(BaseModel):
    """A chunk and how well it matched. Scores are only comparable within one ranking."""

    model_config = ConfigDict(frozen=True)

    chunk: Chunk
    score: float


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunk_id: str = Field(description="The id of the chunk that supports the answer")
    quote: str = Field(
        min_length=1, description="Words copied verbatim from that chunk, without changes"
    )


class ClauseAnswer(BaseModel):
    """The JSON object the model must reply with."""

    model_config = ConfigDict(frozen=True)

    answerable: bool = Field(description="false if the provided chunks don't contain the answer")
    answer: str = Field(description="The answer, using only the chunks, in one or two sentences")
    citations: list[Citation] = Field(
        default_factory=list,
        description="Evidence for the answer: at least one if answerable, none otherwise",
    )


class AnswerResult(BaseModel):
    """A verified answer plus what it took to get it."""

    model_config = ConfigDict(frozen=True)

    answer: ClauseAnswer
    hits: list[SearchHit]
    attempts: int = Field(ge=0, description="Model calls made; 0 = nothing retrieved, no call")
    usage: Usage = Usage()


class AnswerError(Exception):
    """The model never produced a valid, verifiable answer.

    As with ``ExtractionError``, the provider worked but the output was unusable. Provider
    failures stay ``LLMError``.
    """


class EvalQuestion(BaseModel):
    """A question with the section that answers it: the ground truth of retrieval evals."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    doc_id: str
    section: str
    kind: Literal["paraphrase", "exact"] = Field(
        description="paraphrase: different words than the clause; exact: names an identifier"
    )
