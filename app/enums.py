from enum import Enum


class Route(str, Enum):
    SMALLTALK = "smalltalk"
    RESEARCH = "research"


class Intent(str, Enum):
    SMALLTALK = "smalltalk"
    RESEARCH = "research"


class SupportedExtension(str, Enum):
    PDF = ".pdf"
    TXT = ".txt"
    MARKDOWN = ".md"
    MARKDOWN_ALT = ".markdown"
    CSV = ".csv"
    XLS = ".xls"
    XLSX = ".xlsx"
    DOCX = ".docx"


class SSEEvent(str, Enum):
    STATUS = "status"
    TIMING = "timing"
    TOKEN = "token"
    FINAL = "final"
    DONE = "done"
    ERROR = "error"


class Step(str, Enum):
    ROUTER = "router"
    SMALLTALK = "smalltalk"
    PLANNER = "planner"
    WEB_SEARCH = "web_search"
    RETRIEVE = "retrieve"
    WRITER = "writer"
    CRITIC = "critic"
