import json
import re
from collections import deque
from pathlib import Path
from typing import Any
from urllib import error, request
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
KNOWLEDGE_INDEX_FILE = DATA_DIR / "knowledge_index.json"
MAX_STORED_KNOWLEDGE_CHARS = 20000
MAX_PROMPT_KNOWLEDGE_CHARS = 6000
MAX_TOTAL_PROMPT_CHARS = 12000

# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
try:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

TEMPLATE_PATTERN = re.compile(r"{{\s*([A-Za-z0-9_.-]+)\s*}}")
FINAL_RESPONSE_POLICY = (
    "Reply with one direct final answer only. "
    "Do not include analysis, prompt echoes, multiple options, or repeated answers. "
    "Keep the answer short and natural. "
    "If the user asks for code, return only the final useful code."
)

CODE_RESPONSE_POLICY = (
    "The user is asking for code. "
    "Return one complete working code answer only. "
    "Do not explain it. "
    "Do not shorten it to one line. "
    "Do not wrap it in markdown fences unless the user explicitly asks for markdown."
)

MODEL_ALIASES = {
    "gemini": {
        "": "gemini-2.5-flash",
        "gemini 2.5 flash": "gemini-2.5-flash",
        "gemini-2.5-flash": "gemini-2.5-flash",
        "gemini 2.0 flash": "gemini-2.0-flash",
        "gemini-2.0-flash": "gemini-2.0-flash",
        "gemma 4 26b": "gemma-4-26b-a4b-it",
        "gemma-4-26b-a4b-it": "gemma-4-26b-a4b-it",
    },
    "openai": {
        "": "gpt-4.1-mini",
        "gpt-4.1-mini": "gpt-4.1-mini",
        "gpt-4.1": "gpt-4.1",
        "chatgpt": "gpt-4.1-mini",
    },
    "demo": {
        "": "demo-response",
        "demo-response": "demo-response",
    },
}


class PipelinePayload(BaseModel):
    nodes: list[dict[str, Any]] = Field(default_factory=list)
    edges: list[dict[str, Any]] = Field(default_factory=list)


class PipelineRunPayload(PipelinePayload):
    runtime_inputs: dict[str, Any] = Field(default_factory=dict)


def load_knowledge_index() -> dict[str, dict[str, Any]]:
    if not KNOWLEDGE_INDEX_FILE.exists():
        return {}

    try:
        return json.loads(KNOWLEDGE_INDEX_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def save_knowledge_index(index: dict[str, dict[str, Any]]) -> None:
    KNOWLEDGE_INDEX_FILE.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


# KNOWLEDGE_INDEX = load_knowledge_index()
try:
    KNOWLEDGE_INDEX = load_knowledge_index()
except Exception:
    KNOWLEDGE_INDEX = {}

def normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip())


def _clean_line(value: str) -> str:
    return normalize_whitespace(value.strip().strip('"').strip("'"))


def sanitize_context_key(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", value or "").strip("_")
    return normalized or "value"


def truncate_text(value: str, max_chars: int, suffix: str = "\n\n[truncated]") -> str:
    text = value or ""
    if len(text) <= max_chars:
        return text

    cutoff = max(0, max_chars - len(suffix))
    return f"{text[:cutoff].rstrip()}{suffix}"


def is_code_request(prompt: str) -> bool:
    normalized_prompt = (prompt or "").lower()
    return any(
        token in normalized_prompt
        for token in (
            "write code",
            "write a python program",
            "python program",
            "python code",
            "java program",
            "java code",
            "javascript code",
            "code for",
            "script for",
            "function for",
            "program for",
            "program in",
        )
    ) or ("program" in normalized_prompt and any(
        language in normalized_prompt
        for language in ("python", "java", "javascript", "c++", "c ", "c#", "go", "rust")
    ))


def extract_user_query(prompt: str) -> str:
    prompt_text = (prompt or "").replace("\r", "").strip()
    cleaned_lines = []

    for raw_line in prompt_text.splitlines():
        line = _clean_line(raw_line)
        if not line:
            continue

        if line.lower().startswith(("constraint:", "system:", "instruction:", "rules:", "data:")):
            continue

        line = re.sub(
            r"^(question|prompt|input|user asks|user says)\s*:\s*",
            "",
            line,
            flags=re.IGNORECASE,
        )
        cleaned_lines.append(line)

    if cleaned_lines:
        return _clean_line(" ".join(cleaned_lines))

    quoted_items = [
        _clean_line(item)
        for item in re.findall(r'"([^"\n]{2,})"', prompt_text)
        if _clean_line(item)
    ]
    if quoted_items:
        return quoted_items[0]

    return _clean_line(prompt_text)


def build_effective_user_prompt(prompt: str) -> str:
    prompt_text = (prompt or "").replace("\r", "").strip()
    if any(
        token in prompt_text.lower()
        for token in ("question:", "input:", "user asks:", "user says:", "constraint:", "data:")
    ):
        return extract_user_query(prompt_text)

    return prompt_text or extract_user_query(prompt_text)


def build_effective_system_prompt(system_prompt: str, prompt: str = "") -> str:
    base_prompt = (system_prompt or "").strip()
    extra_policy = CODE_RESPONSE_POLICY if is_code_request(prompt) else FINAL_RESPONSE_POLICY
    return f"{base_prompt}\n\n{extra_policy}" if base_prompt else extra_policy


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        if value.get("kind") == "file":
            return value.get("textValue") or value.get("name") or ""
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return "\n".join(to_text(item) for item in value if item is not None)
    return str(value)


def render_template(template: str, context: dict[str, Any]) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        return to_text(context.get(key, ""))

    return TEMPLATE_PATTERN.sub(replace, template or "")


def split_sentences(text: str) -> list[str]:
    return [
        _clean_line(part)
        for part in re.split(r"(?<=[.!?])\s+", text)
        if _clean_line(part)
    ]


def merge_unique_prompt_parts(parts: list[str]) -> str:
    merged_parts: list[str] = []
    seen: set[str] = set()

    for part in parts:
        cleaned_part = (part or "").strip()
        if not cleaned_part:
            continue

        normalized_part = normalize_whitespace(cleaned_part).lower()
        if normalized_part in seen:
            continue

        seen.add(normalized_part)
        merged_parts.append(cleaned_part)

    return "\n\n".join(merged_parts)


def split_prompt_and_knowledge_context(prompt: str) -> tuple[str, str]:
    prompt_text = (prompt or "").replace("\r", "").strip()
    marker = "Knowledge context:"

    if marker not in prompt_text:
        return build_effective_user_prompt(prompt_text), ""

    question_part, knowledge_part = prompt_text.split(marker, 1)
    question = build_effective_user_prompt(question_part.strip())
    knowledge_context = knowledge_part.strip()
    return question, knowledge_context


def extract_section_lines(text: str, heading: str) -> list[str]:
    pattern = re.compile(
        rf"{re.escape(heading)}\n=+\n([\s\S]*?)(?:\n[A-Z][^\n]*\n=+\n|$)"
    )
    match = pattern.search(text)
    if not match:
        return []

    return [line.strip() for line in match.group(1).splitlines() if line.strip()]


def generate_knowledge_fallback_answer(question: str, knowledge_context: str) -> str:
    if not knowledge_context:
        return ""

    normalized_question = (question or "").strip().lower()
    disease_name = re.search(r"^Disease Name:\s*(.+)$", knowledge_context, re.MULTILINE)
    category = re.search(r"^Category:\s*(.+)$", knowledge_context, re.MULTILINE)
    sub_category = re.search(r"^Sub-Category:\s*(.+)$", knowledge_context, re.MULTILINE)
    overview_lines = extract_section_lines(knowledge_context, "Overview")
    symptom_lines = extract_section_lines(knowledge_context, "Symptoms")

    common_symptoms = [
        line.removeprefix("- ").strip()
        for line in symptom_lines
        if line.startswith("- ")
    ][:3]

    disease = disease_name.group(1).strip() if disease_name else "This condition"
    category_text = category.group(1).strip() if category else ""
    sub_category_text = sub_category.group(1).strip() if sub_category else ""
    overview_text = ""

    for line in overview_lines:
        if line and not line.endswith("========"):
            overview_text = line
            break

    if any(token in normalized_question for token in ("what is", "about", "explain")):
        base = f"{disease} is"
        if category_text and sub_category_text:
            base += f" a {category_text.lower()} {sub_category_text.lower()} condition"
        elif category_text:
            base += f" a {category_text.lower()} condition"
        else:
            base += " a medical condition"

        if common_symptoms:
            return f"{base} that commonly causes {', '.join(common_symptoms[:-1])}{' and ' + common_symptoms[-1] if len(common_symptoms) > 1 else common_symptoms[0]}."

        if overview_text:
            return overview_text

        return f"{base}."

    if "symptom" in normalized_question and common_symptoms:
        return f"Common symptoms of {disease.lower()} are {', '.join(common_symptoms[:-1])}{' and ' + common_symptoms[-1] if len(common_symptoms) > 1 else common_symptoms[0]}."

    if "prevent" in normalized_question:
        prevention_lines = [line.removeprefix("- ").strip() for line in extract_section_lines(knowledge_context, "Prevention") if line.startswith("- ")]
        if prevention_lines:
            return f"To help prevent {disease.lower()}, {prevention_lines[0].lower()}."

    if overview_text:
        return overview_text

    summary_lines = [
        line.strip()
        for line in knowledge_context.splitlines()
        if line.strip() and not line.startswith("=") and ":" not in line[:24]
    ]
    if summary_lines:
        return summary_lines[0]

    return ""


def is_prompt_echo(response_text: str, prompt: str) -> bool:
    cleaned_response = _clean_line(response_text).lower()
    cleaned_prompt = _clean_line(build_effective_user_prompt(prompt)).lower()

    if not cleaned_response or not cleaned_prompt:
        return False

    if cleaned_response == cleaned_prompt or cleaned_response.startswith(cleaned_prompt):
        return True

    if len(cleaned_response.split()) <= 3 and cleaned_response in cleaned_prompt:
        return True

    return False


def generate_demo_response(prompt: str) -> str:
    user_query, knowledge_context = split_prompt_and_knowledge_context(prompt)
    normalized_query = user_query.lower()

    knowledge_answer = generate_knowledge_fallback_answer(user_query, knowledge_context)
    if knowledge_answer:
        return knowledge_answer

    if is_code_request(user_query) and "prime" in normalized_query:
        return (
            "def is_prime(num):\n"
            "    if num < 2:\n"
            "        return False\n"
            "    for i in range(2, int(num ** 0.5) + 1):\n"
            "        if num % i == 0:\n"
            "            return False\n"
            "    return True\n\n"
            "num = int(input('Enter a number: '))\n"
            "if is_prime(num):\n"
            "    print(f'{num} is a prime number.')\n"
            "else:\n"
            "    print(f'{num} is not a prime number.')"
        )

    if any(token in normalized_query for token in ("what is your name", "give any name", "tell me a name")):
        return "You can call me Nova."

    if normalized_query.startswith(("hi", "hello", "hey")):
        return "Hello! How can I help you?"

    if normalized_query.endswith("?"):
        return "I can help with that."

    return user_query or "Hello! How can I help you?"


def sanitize_response(response_text: str, prompt: str) -> str:
    cleaned_response = (response_text or "").replace("\r", "").strip()
    effective_prompt = build_effective_user_prompt(prompt)
    question_text, knowledge_context = split_prompt_and_knowledge_context(prompt)

    if not cleaned_response:
        return generate_demo_response(prompt)

    code_block_match = re.search(r"```(?:[A-Za-z0-9#+._-]+)?\n?([\s\S]+?)```", cleaned_response)
    if code_block_match:
        return code_block_match.group(1).strip()

    if is_code_request(effective_prompt):
        if cleaned_response.startswith("```"):
            stripped = re.sub(r"^```[A-Za-z0-9#+._-]*\s*", "", cleaned_response).strip()
            return stripped

        code_lines = [
            line.rstrip()
            for line in cleaned_response.splitlines()
            if line.strip()
        ]
        if code_lines and (
            cleaned_response.startswith("import ")
            or cleaned_response.startswith("public class")
            or any(
                token in cleaned_response
                for token in (
                    "def ",
                    "class ",
                    "for ",
                    "while ",
                    "print(",
                    "return ",
                    "import java",
                    "public static void main",
                    "Scanner",
                    "#include",
                )
            )
        ):
            return "\n".join(code_lines).strip()

    if is_code_request(effective_prompt):
        code_block_match = re.search(r"```(?:python)?\n([\s\S]+?)```", cleaned_response, re.IGNORECASE)
        if code_block_match:
            return code_block_match.group(1).strip()

        code_lines = [
            line.rstrip()
            for line in cleaned_response.splitlines()
            if line.strip()
        ]
        if any(token in cleaned_response for token in ("def ", "for ", "while ", "print(", "return ")):
            return "\n".join(code_lines[:24]).strip()

    filtered_lines = []
    for raw_line in cleaned_response.splitlines():
        line = raw_line.strip(" -*•\t")
        if not line:
            continue

        if line.startswith(
            (
                "Question:",
                "User says:",
                "User asks:",
                "User asks for",
                "Intent:",
                "Option 1",
                "Option 2",
                "Option 3",
                "The user is asking",
                "The system instruction says",
                "Standard response",
                "Response:",
                "Input:",
                "Data:",
                "Constraint:",
                "Wait,",
                "Let's",
            )
        ):
            continue

        filtered_lines.append(_clean_line(line))

    normalized = "\n".join(filtered_lines).strip() or _clean_line(cleaned_response)

    if normalized.lower().startswith("question:"):
        normalized = normalized.split(":", 1)[1].strip()

    if question_text and normalized.lower().startswith(question_text.lower()):
        normalized = normalized[len(question_text):].lstrip(' "\n:-')

    repeated_match = re.match(r'^"?(.{4,}?)"?\s+"?\1"?$', normalized)
    if repeated_match:
        normalized = _clean_line(repeated_match.group(1))

    if is_prompt_echo(normalized, effective_prompt):
        knowledge_answer = generate_knowledge_fallback_answer(question_text, knowledge_context)
        return knowledge_answer or generate_demo_response(prompt)

    if not is_code_request(effective_prompt):
        sentences = split_sentences(normalized)
        if sentences:
            best_sentence = sentences[0]
            if is_prompt_echo(best_sentence, effective_prompt):
                knowledge_answer = generate_knowledge_fallback_answer(question_text, knowledge_context)
                return knowledge_answer or generate_demo_response(prompt)
            return best_sentence

    return normalized or generate_demo_response(prompt)


def normalize_model(provider: str, model: str) -> str:
    normalized_provider = (provider or "demo").lower()
    normalized_model = (model or "").strip()
    normalized_key = normalized_model.lower()
    return MODEL_ALIASES.get(normalized_provider, {}).get(normalized_key, normalized_model)


def get_provider_error_message(status_code: int, detail: str) -> str:
    try:
        payload = json.loads(detail)
    except json.JSONDecodeError:
        payload = {}

    error_payload = payload.get("error", {}) if isinstance(payload, dict) else {}
    message = error_payload.get("message", detail).strip()

    if status_code in {400, 401, 403}:
        return f"Provider rejected the request: {message}"
    if status_code == 404:
        return "The selected model is not available for this provider."
    if status_code == 429:
        return "Provider rate limit reached. Try another model or wait a moment."
    return f"Provider request failed with {status_code}: {message}"


def post_json(url: str, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(get_provider_error_message(exc.code, detail)) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Provider request failed: {exc.reason}") from exc


def extract_gemini_text(payload: dict) -> str:
    try:
        parts = payload["candidates"][0]["content"]["parts"]

        clean_texts = []

        for part in parts:
            # Ignore thinking part
            if part.get("thought"):
                continue

            text = part.get("text", "").strip()
            if text:
                clean_texts.append(text)

        # Return only LAST valid text (best answer)
        return clean_texts[-1] if clean_texts else ""

    except Exception as e:
        print("Extraction error:", e)
        return ""


def extract_openai_text(payload: dict) -> str:
    try:
        message = payload["choices"][0]["message"]
        content = message.get("content", "")

        # Case 1: simple string response
        if isinstance(content, str):
            return content.strip()

        # Case 2: structured content (list)
        if isinstance(content, list):
            clean_parts = []

            for item in content:
                if not isinstance(item, dict):
                    continue

                if item.get("type") in {"text", "output_text"}:
                    text = item.get("text", "").strip()
                    if text:
                        clean_parts.append(text)

            # Return combined clean text
            return "\n".join(clean_parts).strip()

        return ""

    except Exception as e:
        print("OpenAI extraction error:", e)
        return ""


def call_llm(provider: str, model: str, api_key: str, system_prompt: str, prompt: str) -> str:
    normalized_provider = (provider or "demo").lower()
    normalized_model = normalize_model(normalized_provider, model or "")
    effective_system_prompt = build_effective_system_prompt(system_prompt, prompt)
    effective_prompt = build_effective_user_prompt(prompt)
    max_output_tokens = 700 if is_code_request(effective_prompt) else 220

    if normalized_provider == "demo" or not api_key:
        return generate_demo_response(effective_prompt)

    if normalized_provider == "gemini":
        safe_model = quote(normalized_model or "gemini-2.5-flash", safe="-_.")
        response_payload = post_json(
            f"https://generativelanguage.googleapis.com/v1beta/models/{safe_model}:generateContent",
            {
                "systemInstruction": {"parts": [{"text": effective_system_prompt}]},
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": max_output_tokens,
                },
                "contents": [{"role": "user", "parts": [{"text": effective_prompt}]}],
            },
            {
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
        )
        result = extract_gemini_text(response_payload)

        return result

    if normalized_provider == "openai":
        response_payload = post_json(
            "https://api.openai.com/v1/chat/completions",
            {
                "model": normalized_model or "gpt-4.1-mini",
                "temperature": 0.2,
                "max_tokens": max_output_tokens,
                "messages": [
                    {"role": "system", "content": effective_system_prompt},
                    {"role": "user", "content": effective_prompt},
                ],
            },
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        result = extract_openai_text(response_payload)
        return result

    raise RuntimeError("Unsupported provider selected.")


def extract_text_from_upload(file_name: str, raw_bytes: bytes) -> str:
    suffix = Path(file_name).suffix.lower()

    if suffix in {".txt", ".md", ".json", ".csv", ".py", ".js", ".html"}:
        return truncate_text(
            raw_bytes.decode("utf-8", errors="replace").strip(),
            MAX_STORED_KNOWLEDGE_CHARS,
        )

    decoded = raw_bytes.decode("latin-1", errors="ignore")
    snippets = re.findall(r"[A-Za-z0-9 ,.;:!?()\[\]{}\-_'/\\\n]{20,}", decoded)
    joined = "\n".join(snippet.strip() for snippet in snippets[:60]).strip()

    if joined:
        return truncate_text(joined, min(6000, MAX_STORED_KNOWLEDGE_CHARS))

    return f"Uploaded document: {file_name}"


def get_file_context(file_id: str | None) -> str:
    if not file_id:
        return ""

    file_record = KNOWLEDGE_INDEX.get(file_id)
    if not file_record:
        return ""

    return truncate_text(file_record.get("content", ""), MAX_PROMPT_KNOWLEDGE_CHARS)


def clamp_prompt_size(prompt: str) -> str:
    return truncate_text(prompt, MAX_TOTAL_PROMPT_CHARS)


def make_output(default: str = "", **channels: str) -> dict[str, str]:
    output = {"default": to_text(default)}
    output.update({key: to_text(value) for key, value in channels.items()})
    return output


def read_output_channel(value: dict[str, str] | str, source_handle: str | None) -> str:
    if isinstance(value, str):
        return value

    if not source_handle:
        return to_text(value.get("default", ""))

    channel = source_handle.rsplit("-", 1)[-1]
    return to_text(value.get(channel, value.get("default", "")))


def is_directed_acyclic_graph(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> bool:
    node_ids = {node.get("id") for node in nodes if node.get("id") is not None}
    graph = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}

    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")

        if source not in node_ids or target not in node_ids:
            return False

        graph[source].append(target)
        indegree[target] += 1

    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited_count = 0

    while queue:
        node_id = queue.popleft()
        visited_count += 1

        for neighbor in graph[node_id]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return visited_count == len(node_ids)


def topological_sort(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> list[str]:
    node_ids = [node["id"] for node in nodes if node.get("id") is not None]
    graph = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}

    for edge in edges:
        source = edge["source"]
        target = edge["target"]
        graph[source].append(target)
        indegree[target] += 1

    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    ordered = []

    while queue:
        node_id = queue.popleft()
        ordered.append(node_id)

        for neighbor in graph[node_id]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return ordered


def execute_pipeline(pipeline: PipelineRunPayload) -> dict[str, Any]:
    nodes = pipeline.nodes
    edges = pipeline.edges

    if not is_directed_acyclic_graph(nodes, edges):
        raise HTTPException(status_code=400, detail="This pipeline must be a directed acyclic graph.")

    node_map = {node["id"]: node for node in nodes}
    incoming_edges: dict[str, list[dict[str, Any]]] = {node_id: [] for node_id in node_map}

    for edge in edges:
        incoming_edges.setdefault(edge["target"], []).append(edge)

    ordered_node_ids = topological_sort(nodes, edges)
    context: dict[str, Any] = {}
    node_outputs: dict[str, dict[str, str] | str] = {}
    output_values: dict[str, str] = {}
    last_llm_details: dict[str, str] | None = None

    for node_id in ordered_node_ids:
        node = node_map[node_id]
        node_type = node.get("type")
        node_data = node.get("data", {})

        inbound = []
        for edge in incoming_edges.get(node_id, []):
            source_id = edge.get("source")
            if source_id not in node_outputs:
                continue

            inbound.append(
                {
                    "edge": edge,
                    "source_node": node_map.get(source_id, {}),
                    "value": read_output_channel(node_outputs[source_id], edge.get("sourceHandle")),
                }
            )

        upstream_values = [item["value"] for item in inbound if item["value"]]
        upstream_text = "\n\n".join(upstream_values)
        local_context = {**context, "input": upstream_text, "upstream": upstream_text}

        if node_type == "customInput":
            input_name = node_data.get("inputName") or node_id.replace("customInput-", "input_")
            runtime_value = pipeline.runtime_inputs.get(node_id)
            if runtime_value is None:
                runtime_value = node_data.get("inputFile") if node_data.get("inputType") == "file" else node_data.get("inputValue")

            resolved_value = to_text(runtime_value)
            context[input_name] = resolved_value
            node_outputs[node_id] = make_output(resolved_value)
            continue

        if node_type == "text":
            rendered_text = render_template(node_data.get("template", ""), local_context)
            context[sanitize_context_key(node_id)] = rendered_text
            node_outputs[node_id] = make_output(rendered_text)
            continue

        if node_type == "database":
            file_id = node_data.get("selectedFile")
            knowledge_context = get_file_context(file_id)
            file_record = KNOWLEDGE_INDEX.get(file_id or "", {})
            context["knowledge_data"] = knowledge_context
            if file_record.get("name"):
                context[sanitize_context_key(file_record["name"])] = knowledge_context
            node_outputs[node_id] = make_output(knowledge_context)
            continue

        if node_type == "llm":
            prompt_sources = [item["value"] for item in inbound if item["source_node"].get("type") == "text"]
            knowledge_sources = [item["value"] for item in inbound if item["source_node"].get("type") == "database"]
            general_sources = [
                item["value"]
                for item in inbound
                if item["source_node"].get("type") not in {"text", "database"}
            ]

            system_prompt = render_template(node_data.get("systemPrompt", ""), local_context)

            rendered_prompt = render_template(node_data.get("promptTemplate", ""), local_context)
            prompt_parts = [rendered_prompt] if rendered_prompt else []

            if general_sources:
                prompt_parts.extend(general_sources)
            if prompt_sources:
                prompt_parts.extend(prompt_sources)
            if knowledge_sources:
                prompt_parts.append(
                    f"Knowledge context:\n{truncate_text(merge_unique_prompt_parts(knowledge_sources), MAX_PROMPT_KNOWLEDGE_CHARS)}"
                )

            final_prompt = clamp_prompt_size(merge_unique_prompt_parts(prompt_parts) or upstream_text)

            llm_output = call_llm(
                node_data.get("provider", "demo"),
                node_data.get("model", ""),
                node_data.get("apiKey", ""),
                system_prompt,
                final_prompt,
            )
            context["response"] = llm_output
            context[sanitize_context_key(node_id)] = llm_output
            node_outputs[node_id] = make_output(llm_output)
            last_llm_details = {
                "provider": node_data.get("provider", "demo"),
                "model": normalize_model(node_data.get("provider", "demo"), node_data.get("model", "")),
                "prompt": build_effective_user_prompt(final_prompt),
            }
            continue

        if node_type == "customOutput":
            output_name = node_data.get("outputName") or node_id.replace("customOutput-", "output_")
            output_values[output_name] = upstream_text
            node_outputs[node_id] = make_output(upstream_text)
            continue

        node_outputs[node_id] = make_output(upstream_text)

    primary_output = next((value for value in output_values.values() if value), "")

    return {
        "summary": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "is_dag": True,
        },
        "output_text": primary_output,
        "outputs": output_values,
        "llm": last_llm_details,
    }


@app.get("/")
def read_root():
    return {"Ping": "Pong"}


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    raw_bytes = await file.read()
    file_id = uuid4().hex
    file_name = file.filename or f"upload-{file_id}.pdf"
    stored_name = f"{file_id}{Path(file_name).suffix.lower()}"
    stored_path = UPLOAD_DIR / stored_name
    stored_path.write_bytes(raw_bytes)

    extracted_text = extract_text_from_upload(file_name, raw_bytes)
    KNOWLEDGE_INDEX[file_id] = {
        "id": file_id,
        "name": file_name,
        "path": str(stored_path),
        "content": extracted_text,
    }
    save_knowledge_index(KNOWLEDGE_INDEX)

    return {"id": file_id, "name": file_name}


@app.get("/get-files")
def get_files():
    return {
        "files": [
            {"id": file_id, "name": file_info.get("name", file_id)}
            for file_id, file_info in KNOWLEDGE_INDEX.items()
        ]
    }

@app.post("/pipelines/parse")
def parse_pipeline(pipeline: PipelinePayload):
    return {
        "num_nodes": len(pipeline.nodes),
        "num_edges": len(pipeline.edges),
        "is_dag": is_directed_acyclic_graph(pipeline.nodes, pipeline.edges),
    }


@app.post("/pipelines/run")
def run_pipeline(pipeline: PipelineRunPayload):
    try:
        return execute_pipeline(pipeline)
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
