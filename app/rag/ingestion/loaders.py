import json
import logging
from pathlib import Path
from typing import List, TypeVar, Type, Callable, Tuple
from pydantic import BaseModel, ValidationError

from app.rag.ingestion.schemas import FAQ, TechDoc, Troubleshooting, SupportTicket

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

def load_json_files(dir_path: str, model: Type[T]) -> Tuple[List[T], List[str]]:
    path = Path(dir_path)
    valid_items = []
    errors = []
    
    if not path.exists():
        logger.warning(f"Directory not found: {dir_path}")
        return valid_items, errors

    for file_path in path.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            item = model.model_validate(data)
            valid_items.append(item)
        except json.JSONDecodeError as e:
            msg = f"Invalid JSON in {file_path}: {e}"
            logger.error(msg)
            errors.append(msg)
        except ValidationError as e:
            msg = f"Schema validation failed for {file_path}: {e}"
            logger.error(msg)
            errors.append(msg)
        except Exception as e:
            msg = f"Unexpected error reading {file_path}: {e}"
            logger.error(msg)
            errors.append(msg)
            
    return valid_items, errors

def load_faqs(dir_path: str) -> List[FAQ]:
    items, _ = load_json_files(dir_path, FAQ)
    return items

def load_techdocs(dir_path: str) -> List[TechDoc]:
    items, _ = load_json_files(dir_path, TechDoc)
    return items

def load_troubleshooting(dir_path: str) -> List[Troubleshooting]:
    items, _ = load_json_files(dir_path, Troubleshooting)
    return items

def load_tickets(dir_path: str) -> List[SupportTicket]:
    items, _ = load_json_files(dir_path, SupportTicket)
    return items
