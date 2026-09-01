"""Stateless Codex CLI transport using cached ChatGPT authentication."""
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile


_MAX_RESPONSE_BYTES = 512 * 1024
_MODEL_INSTRUCTION = (
    "Act only as the model component of the supplied controller conversation. "
    "Do not inspect the local machine or use tools. Follow system messages as "
    "highest priority and return exactly one JSON object with no Markdown."
)
_CREDENTIAL_VARIABLES = {
    "AGENT" + "_API_KEY",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "CODEX_ACCESS_TOKEN",
}
_MODEL_PATH_VARIABLES = {
    "OLDPWD",
    "PYTHONPATH",
    "GIT_DIR",
    "GIT_WORK_TREE",
}


class CodexCLIError(RuntimeError):
    pass


class CodexCLIClient:
    def __init__(self, executable: str, model: str, prefix_args=(),
                 timeout_seconds=900):
        if not isinstance(executable, str) or not executable.strip():
            raise ValueError("executable must be a non-empty string")
        resolved = shutil.which(executable)
        if resolved is None:
            raise CodexCLIError(f"Codex CLI executable not found: {executable}")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        if isinstance(prefix_args, (str, bytes)):
            raise ValueError("prefix_args must contain string arguments")
        try:
            prefix_args = tuple(prefix_args)
        except TypeError:
            raise ValueError("prefix_args must contain string arguments") from None
        if any(not isinstance(argument, str) for argument in prefix_args):
            raise ValueError("prefix_args must contain string arguments")
        if (isinstance(timeout_seconds, bool)
                or not isinstance(timeout_seconds, (int, float))
                or timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be positive")

        self._command = (resolved, *prefix_args)
        self.model = model
        self.timeout_seconds = timeout_seconds

        version_stdout, _ = self._invoke(
            (*self._command, "--version"),
            failure="version check",
        )
        version_text = self._decode(version_stdout, "version")
        match = re.fullmatch(r"\s*codex-cli\s+(\S+)\s*", version_text)
        if match is None:
            raise CodexCLIError("Codex CLI returned an unrecognized version")

        login_stdout, login_stderr = self._invoke(
            (*self._command, "login", "status"),
            failure="login check",
        )
        login_text = self._decode(
            login_stdout + b"\n" + login_stderr,
            "login status",
        )
        if "Logged in using ChatGPT" not in login_text:
            raise CodexCLIError(
                "Codex CLI must be logged in with ChatGPT; run 'codex login'"
            )
        self.identity = {"version": match.group(1), "model": model}

    @staticmethod
    def _environment(scratch_directory=None):
        environment = os.environ.copy()
        for name in _CREDENTIAL_VARIABLES:
            environment.pop(name, None)
        if scratch_directory is not None:
            for name in _MODEL_PATH_VARIABLES:
                environment.pop(name, None)
            environment["PWD"] = scratch_directory
            environment["TMPDIR"] = scratch_directory
        return environment

    @staticmethod
    def _terminate(process):
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            return process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return process.communicate()

    def _invoke(self, argv, *, failure, input_bytes=None, cwd=None,
                environment=None):
        process = subprocess.Popen(
            list(argv),
            cwd=cwd,
            env=self._environment() if environment is None else environment,
            shell=False,
            stdin=subprocess.PIPE if input_bytes is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(
                input=input_bytes,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            self._terminate(process)
            raise CodexCLIError(
                f"Codex CLI timed out after {self.timeout_seconds} seconds"
            ) from None
        except BaseException:
            self._terminate(process)
            raise
        if process.returncode:
            if failure == "model call":
                raise CodexCLIError(
                    f"Codex CLI exited with status {process.returncode}"
                )
            raise CodexCLIError(
                f"Codex CLI {failure} failed with exit status {process.returncode}"
            )
        return stdout, stderr

    @staticmethod
    def _decode(value, context):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            raise CodexCLIError(
                f"Codex CLI {context} output is not valid UTF-8"
            ) from None

    @staticmethod
    def _validate_request(messages, request_id):
        if not isinstance(request_id, str) or not request_id.strip():
            raise ValueError("request_id must be a non-empty string")
        if not isinstance(messages, list):
            raise ValueError("messages must be a list")
        for message in messages:
            if (not isinstance(message, dict)
                    or set(message) != {"role", "content"}
                    or not isinstance(message["role"], str)
                    or not isinstance(message["content"], str)):
                raise ValueError(
                    "messages must contain only role/content string records"
                )

    @staticmethod
    def _parse_response(stdout):
        if len(stdout) > _MAX_RESPONSE_BYTES:
            raise CodexCLIError("Codex CLI response exceeds 512 KiB")
        text = CodexCLIClient._decode(stdout, "response")
        stripped = text.lstrip()
        try:
            value, end = json.JSONDecoder().raw_decode(stripped)
        except json.JSONDecodeError:
            raise CodexCLIError("Codex CLI response is malformed JSON") from None
        if stripped[end:].strip():
            raise CodexCLIError("Codex CLI response contains trailing content")
        if not isinstance(value, dict):
            raise CodexCLIError("Codex CLI response must be a JSON object")
        return value

    def complete(self, messages, request_id):
        self._validate_request(messages, request_id)
        prompt = json.dumps(
            {
                "instruction": _MODEL_INSTRUCTION,
                "request_id": request_id,
                "messages": messages,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        with tempfile.TemporaryDirectory(prefix="kuairand-codex-") as scratch:
            stdout, _ = self._invoke(
                (
                    *self._command,
                    "exec",
                    "--sandbox",
                    "read-only",
                    "--skip-git-repo-check",
                    "--model",
                    self.model,
                    "--color",
                    "never",
                    "-",
                ),
                failure="model call",
                input_bytes=prompt,
                cwd=scratch,
                environment=self._environment(scratch),
            )
        return self._parse_response(stdout)
